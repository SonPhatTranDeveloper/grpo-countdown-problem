from collections.abc import Callable

from datasets import Dataset, load_dataset
from openai import OpenAI


def map_problem_description_to_conversation_grpo(
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
You are an expert mathematician specializing in arithmetic countdown problems. Your task is to find arithmetic expressions using exactly four given numbers and basic operators (+, -, *, /) to reach a target result.

**Your approach must be:**
1. Use **a single <think> block** to show your systematic reasoning process
2. Consider different combinations of numbers and operators
3. Apply proper order of operations (multiplication and division before addition and subtraction)
4. Verify your calculations step by step
5. Provide your final arithmetic expression in the <answer> block
6. There should ONLY be ONE <answer> block containing only the arithmetic expression.

**Rules:**
- Use each of the four given numbers exactly once
- Only use operators: +, -, *, / (use '*' for multiplication)
- The expression must equal the target result exactly
- Show clear mathematical reasoning in your thinking
- Your final answer must be a valid arithmetic expression

**Format:**
<think>
Analyze the numbers and target result, try different combinations and operations, calculate and verify results step by step.
</think>
<answer>
(Your arithmetic expression, e.g., "3 + 7 * 2 - 1")
</answer>

Example:
<think>
Analyze the numbers and target result, try different combinations and operations, calculate and verify results step by step.
</think>
<answer>
3 + 7 * 2 - 1
</answer>

There should ONLY be ONE <answer> block containing only the arithmetic expression.
"""
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": row["problem_description"]},
        ]
    }


def get_reasoning_for_answer(problem_description: str) -> str:
    """
    Get the reasoning for the answer using OpenAI GPT-4o-mini.

    Args:
        problem_description: The problem description

    Returns:
        str: The reasoning for the answer in <think>...</think><answer>...</answer> format
    """
    client = OpenAI()

    system_prompt = """You are an expert mathematician specializing in arithmetic countdown problems. Your task is to find arithmetic expressions using exactly four given numbers and basic operators (+, -, *, /) to reach a target result.

**Your approach must be:**
1. Use **a single <think> block** to show your systematic reasoning process
2. Consider different combinations of numbers and operators
3. Apply proper order of operations (multiplication and division before addition and subtraction)
4. Verify your calculations step by step
5. Provide your final arithmetic expression in the <answer> block
6. There should ONLY be ONE <answer> block containing only the arithmetic expression.

**Rules:**
- Must use ALL the given numbers in the arithmetic expression
- Only use operators: +, -, *, /
- The expression must equal the target result exactly
- Don't use parenthesis in the arithmetic expression
- Reasoning should be clear and detailed, but not too verbose (aim for 100-200 words)
- Your final answer must be a valid arithmetic expression

**Answer Format:**
<think>
Analyze the numbers and target result, try different combinations and operations, calculate and verify results step by step.
</think>
<answer>
Your arithmetic expression, e.g., 3 + 7 * 2 - 1
</answer>

There should ONLY be ONE <answer> block containing only the arithmetic expression."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem_description},
        ],
        max_tokens=2048,
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()


def load_csv_dataset_grpo(
    file_path: str, split: str, mapping_function: Callable
) -> Dataset:
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
