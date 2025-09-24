from collections.abc import Callable

from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from openai import OpenAI

from src.utils.caching import cached


def map_problem_description_to_conversation(
    row: dict[str, any],
) -> list[dict[str, any]]:
    """
    Map a problem description to a conversation.

    Args:
        row: The row

    Returns:
        list[dict[str, any]]: The conversation
    """
    system_prompt = """
You are an expert mathematician specializing in arithmetic countdown problems. Your task is to find arithmetic expressions using exactly four given numbers and basic operators (+, -, x, /) to reach a target result.

**Your approach must be:**
1. Use **a single <think> block** to show your systematic reasoning process
2. Consider different combinations of numbers and operators
3. Apply proper order of operations (multiplication and division before addition and subtraction)
4. Verify your calculations step by step
5. Provide your final arithmetic expression in the <answer> block
6. There should ONLY be ONE <answer> block containing only the arithmetic expression.

**Rules:**
- Use each of the four given numbers exactly once
- Only use operators: +, -, x, / (use 'x' for multiplication, not '*')
- The expression must equal the target result exactly
- Show clear mathematical reasoning in your thinking
- Your final answer must be a valid arithmetic expression

**Format:**
<think>
Analyze the numbers and target result, try different combinations and operations, calculate and verify results step by step.
</think>
<answer>
(Your arithmetic expression, e.g., "3 + 7 x 2 - 1")
</answer>

There should ONLY be ONE <answer> block containing only the arithmetic expression.
"""
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": row["problem_description"]},
        ]
    }


@cached(cache_dir="cache")
def get_reasoning_for_answer(problem_description: str) -> str:
    """
    Get the reasoning for the answer using OpenAI GPT-4o-mini.

    Args:
        problem_description: The problem description

    Returns:
        str: The reasoning for the answer in <think>...</think><answer>...</answer> format
    """
    client = OpenAI()

    system_prompt = """You are an expert mathematician specializing in arithmetic countdown problems. Your task is to find arithmetic expressions using exactly four given numbers and basic operators (+, -, x, /) to reach a target result.

**Your approach must be:**
1. Use **a single <think> block** to show your systematic reasoning process
2. Consider different combinations of numbers and operators
3. Apply proper order of operations (multiplication and division before addition and subtraction)
4. Verify your calculations step by step
5. Provide your final arithmetic expression in the <answer> block
6. There should ONLY be ONE <answer> block containing only the arithmetic expression.

**Rules:**
- Use each of the four given numbers exactly once
- Only use operators: +, -, x, / (use 'x' for multiplication, not '*')
- The expression must equal the target result exactly
- Show clear mathematical reasoning in your thinking
- Your final answer must be a valid arithmetic expression

**Answer Format:**
<think>
Analyze the numbers and target result, try different combinations and operations, calculate and verify results step by step.
</think>
<answer>
Your arithmetic expression, e.g., 3 + 7 x 2 - 1
</answer>

There should ONLY be ONE <answer> block containing only the arithmetic expression."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem_description},
        ],
        temperature=1.0,
        max_tokens=1000,
    )

    return response.choices[0].message.content.strip()


def load_csv_dataset(file_path: str, split: str, mapping_function: Callable) -> Dataset:
    """
    Load a CSV dataset.

    Args:
        file_path: Path to the CSV file
        mapping_function: Function to map the dataset
        split: Split of the dataset

    Returns:
        Dataset: The loaded dataset
    """
    dataset = load_dataset("csv", data_files=file_path, split=split)
    dataset = dataset.map(mapping_function)
    return dataset


def load_huggingface_dataset(
    dataset_name: str, split: str, mapping_function: Callable
) -> Dataset:
    """
    Load a Hugging Face dataset.

    Args:
        dataset_name: Name of the dataset
        split: Split of the dataset
        mapping_function: Function to map the dataset

    Returns:
        Dataset: The loaded dataset
    """
    dataset = load_dataset(dataset_name, split=split)
    dataset = dataset.map(mapping_function)
    return dataset


if __name__ == "__main__":
    load_dotenv()
    print(
        get_reasoning_for_answer(
            "Using the numbers 1, 4, 7, and 8, create an expression that equals 61. You can only use +, -, x, and / operators."
        )
    )
