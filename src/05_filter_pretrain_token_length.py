import json
from typing import Dict, List

import sentencepiece as spm

from config import (
    PRETRAIN_METHODS_JSONL,
    PROCESSED_DIR,
    TOKENIZER_MODEL_PATH,
    MIN_TOKENS,
    MAX_TOKENS,
)


FILTERED_PRETRAIN_JSONL = PROCESSED_DIR / "pretrain_methods_filtered.jsonl"
FILTERED_PRETRAIN_TXT = PROCESSED_DIR / "pretrain_methods_filtered.txt"


def load_methods() -> List[Dict[str, str]]:
    """Load the raw pretraining methods saved earlier."""
    methods = []

    with open(PRETRAIN_METHODS_JSONL, "r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            methods.append(item)

    return methods


def token_length(sp: spm.SentencePieceProcessor, text: str) -> int:
    """Return the number of SentencePiece tokens in a method."""
    return len(sp.encode(text, out_type=int))


def save_filtered(methods: List[Dict[str, str]]) -> None:
    """Save filtered methods as JSONL and TXT."""
    with open(FILTERED_PRETRAIN_JSONL, "w", encoding="utf-8") as jsonl_file:
        for item in methods:
            jsonl_file.write(json.dumps(item) + "\n")

    with open(FILTERED_PRETRAIN_TXT, "w", encoding="utf-8") as txt_file:
        for item in methods:
            single_line = item["method"].replace("\n", " ")
            txt_file.write(single_line + "\n")


def main() -> None:
    """Filter pretraining methods by tokenizer length."""
    print("Loading tokenizer...")
    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_MODEL_PATH))

    print("Loading raw methods...")
    methods = load_methods()

    print(f"Raw method count: {len(methods)}")

    filtered = []
    lengths = []

    for item in methods:
        method_text = item["method"]
        length = token_length(sp, method_text)

        if MIN_TOKENS <= length <= MAX_TOKENS:
            filtered.append(
                {
                    "method": method_text,
                    "token_length": length,
                }
            )
            lengths.append(length)

    print(f"Filtered method count: {len(filtered)}")

    if lengths:
        print(f"Minimum token length: {min(lengths)}")
        print(f"Maximum token length: {max(lengths)}")
        print(f"Average token length: {sum(lengths) / len(lengths):.2f}")

    print("Saving filtered methods...")
    save_filtered(filtered)

    print(f"Saved JSONL to: {FILTERED_PRETRAIN_JSONL}")
    print(f"Saved TXT to: {FILTERED_PRETRAIN_TXT}")


if __name__ == "__main__":
    main()