#!/usr/bin/env python3
"""
GRPO training script for arithmetic countdown problems using Hydra configuration.

This script trains a language model using GRPO (Group Relative Policy Optimization)
to solve arithmetic problems with proper reasoning and formatting.
"""

import logging
import os
from collections.abc import Callable
from pathlib import Path

import hydra
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, PreTrainedModel
from trl import GRPOConfig, GRPOTrainer

from src.dataset import load_csv_dataset
from src.dataset.grpo import map_problem_description_to_conversation
from src.utils.rewards import (
    arithmetic_format_reward_function,
    correctness_reward_function,
    format_reward_functiondef,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("grpo_training")


def load_train_dataset(cfg: DictConfig) -> Dataset:
    """
    Load, shuffle, and subsample the training dataset.

    Args:
        cfg: Dataset configuration

    Returns:
        Dataset: A datasets.Dataset ready for GRPO training
    """
    raw_dataset: Dataset = load_csv_dataset(
        cfg.file_path, cfg.split, map_problem_description_to_conversation
    )
    raw_dataset = raw_dataset.shuffle(seed=cfg.seed)
    train_dataset = raw_dataset.select(range(min(cfg.max_rows, len(raw_dataset))))
    logger.info("Train rows: %d", len(train_dataset))
    return train_dataset


def create_lora_model(cfg: DictConfig) -> PreTrainedModel:
    """
    Create a base causal LM and wrap it with LoRA adapters.

    Args:
        cfg: Model configuration

    Returns:
        PreTrainedModel: A transformers.PreTrainedModel with LoRA adapters applied
    """
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        device_map=cfg.device_map,
    )

    lora_cfg = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        target_modules=cfg.lora.target_modules,
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
        task_type=cfg.lora.task_type,
    )
    model = get_peft_model(model, lora_cfg)
    logger.info("Model with LoRA ready")
    return model


def create_grpo_config(cfg: DictConfig, output_dir: str) -> GRPOConfig:
    """
    Create GRPO training configuration from Hydra config.

    Args:
        cfg: Training configuration
        output_dir: Directory where checkpoints and logs will be written

    Returns:
        GRPOConfig: A configured trl.GRPOConfig instance
    """
    return GRPOConfig(
        output_dir=output_dir,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        optim=cfg.optim,
        remove_unused_columns=cfg.remove_unused_columns,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        bf16=cfg.bf16,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        temperature=cfg.temperature,
        # Preprocessing controls
        max_completion_length=cfg.max_completion_length,
        num_generations=cfg.num_generations,
        max_prompt_length=cfg.max_prompt_length,
        # Logging and saving
        report_to=cfg.report_to,
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
    )


def create_trainer(
    model: PreTrainedModel,
    train_dataset: Dataset,
    args: GRPOConfig,
) -> GRPOTrainer:
    """
    Construct a GRPOTrainer with arithmetic-specific reward functions.

    Args:
        model: The LoRA-wrapped pretrained model to train
        train_dataset: The dataset to use for training
        args: The GRPO configuration

    Returns:
        GRPOTrainer: An initialized trl.GRPOTrainer instance
    """
    reward_funcs: list[Callable[..., list[float]]] = [
        format_reward_functiondef,  # Checks for <think> and <answer> tags
        arithmetic_format_reward_function,  # Validates arithmetic expression format
        correctness_reward_function,  # Rewards based on correctness of the answer
    ]
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=args,
        train_dataset=train_dataset,
    )
    return trainer


def train_and_save(trainer: GRPOTrainer, output_dir: str) -> None:
    """
    Run training and save the final model to disk.

    Args:
        trainer: The configured GRPO trainer instance
        output_dir: Output directory to save the trained model

    Returns:
        None
    """
    train_result = trainer.train()
    logger.info("Training complete: %s", str(train_result))
    trainer.save_model(output_dir)
    logger.info("Saved to %s", output_dir)


@hydra.main(version_base=None, config_path="../../config/grpo", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Run the full GRPO training workflow with Hydra configuration.

    Args:
        cfg: Hydra configuration object

    Returns:
        None
    """
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    # Validate dataset file exists
    if not Path(cfg.dataset.file_path).exists():
        logger.error("Dataset CSV file does not exist: %s", cfg.dataset.file_path)
        return

    if cfg.dataset.max_rows <= 0:
        logger.error("max_rows must be positive")
        return

    # Create output directory
    os.makedirs(cfg.output_dir, exist_ok=True)
    logger.info("Output dir: %s", cfg.output_dir)

    # Load dataset
    train_dataset = load_train_dataset(cfg.dataset)

    # Create model
    model = create_lora_model(cfg.model)

    # Create training configuration
    training_args = create_grpo_config(cfg.training, cfg.output_dir)

    # Create trainer
    trainer = create_trainer(
        model=model, train_dataset=train_dataset, args=training_args
    )

    # Train and save
    train_and_save(trainer=trainer, output_dir=cfg.output_dir)


if __name__ == "__main__":
    main()
