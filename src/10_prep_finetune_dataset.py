import json
from typing import Dict, Any, List

import sentencepiece as spm
from datasets import load_dataset
from tqdm import tqdm

from config import (
    TOKENIZER_MODEL_PATH,
    PROCESSED_DIR,
)


FINETUNE_TRAIN_JSONL = PROCESSED_DIR / "finetune_train.jsonl"
FINETUNE_VALID_JSONL = PROCESSED_DIR / "finetune_valid.jsonl"
FINETUNE_TEST_JSONL = PROCESSED_DIR / "finetune_test.jsonl"

MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 512


def encode_example(
    sp: spm.SentencePieceProcessor,
    item: Dict[str, Any],
) -> Dict[str, Any]:
    """Convert one bug-fixing example into token IDs."""
    buggy_code = item["buggy"].strip()
    fixed_code = item["fixed"].strip()

    input_ids = sp.encode(buggy_code, out_type=int)[:MAX_INPUT_LENGTH]
    labels = sp.encode(fixed_code, out_type=int)[:MAX_TARGET_LENGTH]

    # Add EOS token to target if missing
    if len(labels) == 0 or labels[-1] != sp.eos_id():
        labels.append(sp.eos_id())

    return {
        "buggy": buggy_code,
        "fixed": fixed_code,
        "input_ids": input_ids,
        "labels": labels,
    }


def save_split(
    split_data,
    output_path,
    sp: spm.SentencePieceProcessor,
    split_name: str,
) -> None:
    """Tokenize and save one dataset split."""
    print(f"Processing {split_name} split...")
    count = 0

    with open(output_path, "w", encoding="utf-8") as file:
        for item in tqdm(split_data):
            example = encode_example(sp, item)
            file.write(json.dumps(example) + "\n")
            count += 1

    print(f"Saved {count} examples to: {output_path}")


def main() -> None:
    """Prepare CodeXGLUE bug-fixing dataset for fine-tuning."""
    print("Loading tokenizer...")
    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_MODEL_PATH))

    print("Loading CodeXGLUE code refinement dataset...")
    dataset = load_dataset(
        "google/code_x_glue_cc_code_refinement",
        name="medium",
    )

    print(dataset)

    save_split(dataset["train"], FINETUNE_TRAIN_JSONL, sp, "train")
    save_split(dataset["validation"], FINETUNE_VALID_JSONL, sp, "validation")
    save_split(dataset["test"], FINETUNE_TEST_JSONL, sp, "test")

    print("\nFine-tuning dataset preparation complete.")


if __name__ == "__main__":
    main()