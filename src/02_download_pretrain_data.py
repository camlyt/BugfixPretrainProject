import json
import random
from typing import List, Dict, Any

from datasets import load_dataset

from config import (
    SEED,
    PRETRAIN_SAMPLE_SIZE,
    PRETRAIN_METHODS_TXT,
    PRETRAIN_METHODS_JSONL,
)


def clean_method_text(text: str) -> str:
    """Basic cleanup for a Java method string."""
    return text.strip()


def extract_methods(records: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Extract method text from CodeSearchNet records.

    The assignment says to use the whole_func_string field.
    """
    extracted = []

    for record in records:
        method_text = record.get("whole_func_string", "")
        method_text = clean_method_text(method_text)

        if method_text:
            extracted.append({"method": method_text})

    return extracted


def save_txt(methods: List[Dict[str, str]]) -> None:
    """Save one method per line for tokenizer training."""
    with open(PRETRAIN_METHODS_TXT, "w", encoding="utf-8") as file:
        for item in methods:
            single_line = item["method"].replace("\n", " ")
            file.write(single_line + "\n")


def save_jsonl(methods: List[Dict[str, str]]) -> None:
    """Save methods in JSONL for easier later inspection."""
    with open(PRETRAIN_METHODS_JSONL, "w", encoding="utf-8") as file:
        for item in methods:
            file.write(json.dumps(item) + "\n")


def main() -> None:
    """Download and save the Java pretraining corpus sample."""
    random.seed(SEED)

    print("Loading CodeSearchNet Java dataset...")
    dataset = load_dataset("code_search_net", "java")

    print("Shuffling and sampling training split...")
    train_split = dataset["train"].shuffle(seed=42).select(range(50000))
    sampled = train_split.select(range(PRETRAIN_SAMPLE_SIZE))

    print("Converting sampled records to a Python list...")
    sampled_records = [sampled[i] for i in range(len(sampled))]

    print("Extracting method text...")
    methods = extract_methods(sampled_records)

    print(f"Extracted {len(methods)} methods.")

    print("Saving text file for tokenizer training...")
    save_txt(methods)

    print("Saving JSONL copy for inspection...")
    save_jsonl(methods)

    print(f"Saved tokenizer text corpus to: {PRETRAIN_METHODS_TXT}")
    print(f"Saved JSONL corpus to: {PRETRAIN_METHODS_JSONL}")


if __name__ == "__main__":
    main()