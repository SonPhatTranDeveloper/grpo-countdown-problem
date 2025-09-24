from datasets import Dataset, load_dataset

SYSTEM_PROMPT = """
You are a helpful AI Assistant that follows user's instructions.
- For every user query, you will first reason through your solution and thought process. You are encouraged to use **multiple <think> blocks** to detail your step-by-step thinking.
- After your reasoning, provide your final response to the user. This response must be contained within <answer> tags.
- Your entire output must consist *only* of these blocks, one after the other, with no additional text or explanations outside of them.

**Example Format:**
<think>
(First step of reasoning)
</think>
<think>
(Second step of reasoning)
</think>
<think>
(Many more steps of reasoning)
</think>
<answer>
(Your final, helpful response here.)
</answer>

There should ONLY be ONE <answer> block.
"""


def map_proble_description_to_conversation(
    row: dict[str, any],
) -> list[dict[str, any]]:
    """
    Map a problem description to a conversation.

    Args:
        row: The row

    Returns:
        list[dict[str, any]]: The conversation
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": row["problem_description"]},
    ]


def load_csv_dataset(file_path: str) -> Dataset:
    """
    Load a CSV dataset.

    Args:
        file_path: Path to the CSV file

    Returns:
        Dataset: The loaded dataset
    """
    dataset = load_dataset("csv", data_files=file_path)
    dataset = dataset.map(map_proble_description_to_conversation)
    return dataset
