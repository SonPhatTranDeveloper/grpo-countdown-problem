#!/usr/bin/env python3
"""
Script to calculate accuracy of a trained GRPO model on arithmetic countdown problems.

This script loads a CSV file with problem data, performs inference using the trained model,
and calculates the accuracy by comparing predicted answers with correct answers.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.inference import GRPOModelInference

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("calculate_accuracy")


def load_csv_data(csv_path: str) -> pd.DataFrame:
    """
    Load CSV data with the expected format.

    Expected columns: id, problem_description, correct_answer, num1, num2, num3, num4

    Args:
        csv_path: Path to the CSV file

    Returns:
        DataFrame with the loaded data
    """
    df = pd.read_csv(csv_path)

    # Verify required columns exist
    required_columns = [
        "id",
        "problem_description",
        "correct_answer",
        "num1",
        "num2",
        "num3",
        "num4",
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    logger.info(f"Loaded {len(df)} problems from {csv_path}")
    return df


def safe_eval_expression(expression: str) -> tuple[float | None, bool]:
    """
    Safely evaluate an arithmetic expression.

    Args:
        expression: The arithmetic expression to evaluate

    Returns:
        Tuple of (result, is_valid)
    """
    if not expression or not expression.strip():
        return None, False

    # Replace 'x' with '*' for evaluation if present
    normalized = expression.replace("x", "*").replace("X", "*")

    # Basic validation - only allow numbers, operators, spaces, and parentheses
    allowed_chars = set("0123456789+-*/.() ")
    if not all(c in allowed_chars for c in normalized):
        return None, False

    try:
        result = eval(normalized)
        return result, True
    except (SyntaxError, ValueError, ZeroDivisionError, NameError):
        return None, False


def evaluate_prediction(
    predicted_answer: str, correct_answer: str, nums: list[int]
) -> dict:
    """
    Evaluate a single prediction against the correct answer.

    Args:
        predicted_answer: The model's predicted arithmetic expression
        correct_answer: The correct arithmetic expression
        nums: List of four numbers used in the problem

    Returns:
        Dictionary with evaluation results
    """
    result = {
        "predicted_answer": predicted_answer,
        "correct_answer": correct_answer,
        "is_correct": False,
        "is_valid_format": False,
        "predicted_result": None,
        "correct_result": None,
    }

    # Evaluate predicted answer
    predicted_result, is_valid_predicted = safe_eval_expression(predicted_answer)
    result["predicted_result"] = predicted_result
    result["is_valid_format"] = is_valid_predicted

    # Evaluate correct answer
    correct_result, is_valid_correct = safe_eval_expression(correct_answer)
    result["correct_result"] = correct_result

    # Check if prediction is correct
    if (
        is_valid_predicted
        and is_valid_correct
        and predicted_result is not None
        and correct_result is not None
    ):
        result["is_correct"] = abs(predicted_result - correct_result) < 1e-6

    return result


def calculate_accuracy(
    csv_path: str,
    sft_model_path: str,
    grpo_model_path: str,
    base_model_id: str = "Qwen/Qwen2.5-Math-1.5B",
    device: str = "auto",
    dtype: torch.dtype = torch.float16,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    max_samples: int | None = None,
    output_path: str | None = None,
) -> dict:
    """
    Calculate accuracy of the model on the given dataset.

    Args:
        csv_path: Path to the CSV file with test data
        sft_model_path: Path to the SFT model
        grpo_model_path: Path to the GRPO model
        base_model_id: Base model identifier
        device: Device to run inference on
        dtype: Data type for the model
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        max_samples: Maximum number of samples to evaluate (None for all)
        output_path: Path to save detailed results (optional)

    Returns:
        Dictionary with accuracy metrics
    """
    # Load data
    df = load_csv_data(csv_path)

    if max_samples is not None:
        df = df.head(max_samples)
        logger.info(f"Limiting evaluation to {max_samples} samples")

    # Initialize model
    logger.info("Loading model...")
    model_inference = GRPOModelInference(
        sft_model_path=sft_model_path,
        grpo_model_path=grpo_model_path,
        base_model_id=base_model_id,
        device=device,
        dtype=dtype,
    )

    # Evaluate each problem
    results = []
    correct_predictions = 0
    valid_format_predictions = 0

    logger.info("Starting evaluation...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        # Perform inference
        response, extracted_answer, _ = model_inference.solve_problem(
            problem_description=row["problem_description"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        # Evaluate prediction
        nums = [row["num1"], row["num2"], row["num3"], row["num4"]]
        evaluation = evaluate_prediction(
            predicted_answer=extracted_answer,
            correct_answer=row["correct_answer"],
            nums=nums,
        )

        # Add metadata
        evaluation.update(
            {
                "id": row["id"],
                "problem_description": row["problem_description"],
                "full_response": response,
                "nums": nums,
            }
        )

        results.append(evaluation)

        # Update counters
        if evaluation["is_correct"]:
            correct_predictions += 1
        if evaluation["is_valid_format"]:
            valid_format_predictions += 1

    # Calculate metrics
    total_samples = len(results)
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    valid_format_rate = (
        valid_format_predictions / total_samples if total_samples > 0 else 0
    )

    metrics = {
        "total_samples": total_samples,
        "correct_predictions": correct_predictions,
        "valid_format_predictions": valid_format_predictions,
        "accuracy": accuracy,
        "valid_format_rate": valid_format_rate,
    }

    # Log results
    logger.info("Evaluation completed!")
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Correct predictions: {correct_predictions}")
    logger.info(f"Valid format predictions: {valid_format_predictions}")
    logger.info(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    logger.info(
        f"Valid format rate: {valid_format_rate:.4f} ({valid_format_rate * 100:.2f}%)"
    )

    # Save detailed results if requested
    if output_path:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path, index=False)
        logger.info(f"Detailed results saved to {output_path}")

    return metrics


def main():
    """Main function to run the accuracy calculation script."""
    parser = argparse.ArgumentParser(
        description="Calculate accuracy of GRPO model on arithmetic countdown problems"
    )

    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        default="data/grpo/test.csv",
        help="Path to CSV file with test data",
    )
    parser.add_argument(
        "--sft_model_path",
        type=str,
        default="models/sft/",
        help="Path to SFT model directory",
    )
    parser.add_argument(
        "--grpo_model_path",
        type=str,
        default="models/grpo/",
        help="Path to GRPO model directory",
    )
    parser.add_argument(
        "--base_model_id",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B",
        help="Base model identifier",
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to run inference on"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=512, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save detailed results CSV",
    )

    args = parser.parse_args()

    # Convert dtype
    dtype = torch.float16

    # Calculate accuracy
    metrics = calculate_accuracy(
        csv_path=args.csv_path,
        sft_model_path=args.sft_model_path,
        grpo_model_path=args.grpo_model_path,
        base_model_id=args.base_model_id,
        device=args.device,
        dtype=dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        max_samples=args.max_samples,
        output_path=args.output_path,
    )

    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(
        f"Valid Format Rate: {metrics['valid_format_rate']:.4f} ({metrics['valid_format_rate'] * 100:.2f}%)"
    )
    print(
        f"Correct Predictions: {metrics['correct_predictions']}/{metrics['total_samples']}"
    )
    print("=" * 50)


if __name__ == "__main__":
    main()
