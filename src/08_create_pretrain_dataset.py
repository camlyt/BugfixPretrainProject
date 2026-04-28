import json
import random
from typing import Dict, List, Tuple

import sentencepiece as spm
from tqdm import tqdm

from config import (
    TOKENIZER_MODEL_PATH,
    PROCESSED_DIR,
    SEED,
)


FILTERED_PRETRAIN_JSONL = PROCESSED_DIR / "pretrain_methods_filtered.jsonl"
PRETRAIN_SPAN_CORRUPTION_JSONL = PROCESSED_DIR / "pretrain_span_corruption.jsonl"

CORRUPTION_RATE = 0.15


def get_sentinel_id(sp: spm.SentencePieceProcessor, index: int) -> int:
    """Return the token ID for <extra_id_index>."""
    return sp.piece_to_id(f"<extra_id_{index}>")


def choose_mask_positions(num_tokens: int, corruption_rate: float) -> List[int]:
    """Choose token positions to mask using short contiguous spans."""
    num_to_mask = max(1, round(num_tokens * corruption_rate))
    masked_positions = set()

    while len(masked_positions) < num_to_mask:
        start = random.randint(0, num_tokens - 1)
        span_length = random.randint(1, 3)

        for pos in range(start, min(start + span_length, num_tokens)):
            if len(masked_positions) < num_to_mask:
                masked_positions.add(pos)

    return sorted(masked_positions)


def group_consecutive_positions(positions: List[int]) -> List[List[int]]:
    """Group consecutive mask positions into spans."""
    if not positions:
        return []

    spans = [[positions[0]]]

    for pos in positions[1:]:
        if pos == spans[-1][-1] + 1:
            spans[-1].append(pos)
        else:
            spans.append([pos])

    return spans


def apply_span_corruption(
    sp: spm.SentencePieceProcessor,
    token_ids: List[int],
) -> Tuple[List[int], List[int]]:
    """Create corrupted input IDs and target IDs using T5-style sentinel tokens."""
    mask_positions = choose_mask_positions(len(token_ids), CORRUPTION_RATE)
    spans = group_consecutive_positions(mask_positions)

    corrupted_input = []
    target = []

    span_lookup = {}
    for span_index, span in enumerate(spans):
        for pos in span:
            span_lookup[pos] = span_index

    i = 0
    while i < len(token_ids):
        if i in span_lookup:
            span_index = span_lookup[i]
            span = spans[span_index]

            sentinel_id = get_sentinel_id(sp, span_index)

            corrupted_input.append(sentinel_id)

            target.append(sentinel_id)
            for pos in span:
                target.append(token_ids[pos])

            i = span[-1] + 1
        else:
            corrupted_input.append(token_ids[i])
            i += 1

    target.append(sp.eos_id())

    return corrupted_input, target


def load_filtered_methods() -> List[Dict[str, str]]:
    """Load filtered pretraining methods."""
    methods = []

    with open(FILTERED_PRETRAIN_JSONL, "r", encoding="utf-8") as file:
        for line in file:
            methods.append(json.loads(line))

    return methods


def main() -> None:
    """Create the span-corruption pretraining dataset."""
    random.seed(SEED)

    print("Loading tokenizer...")
    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_MODEL_PATH))

    print("Loading filtered methods...")
    methods = load_filtered_methods()
    print(f"Loaded {len(methods)} methods.")

    print("Creating span-corruption examples...")

    with open(PRETRAIN_SPAN_CORRUPTION_JSONL, "w", encoding="utf-8") as out_file:
        for item in tqdm(methods):
            method_text = item["method"]
            original_ids = sp.encode(method_text, out_type=int)

            input_ids, labels = apply_span_corruption(sp, original_ids)

            example = {
                "input_text": sp.decode(input_ids),
                "target_text": sp.decode(labels),
                "input_ids": input_ids,
                "labels": labels,
            }

            out_file.write(json.dumps(example) + "\n")

    print(f"Saved pretraining examples to: {PRETRAIN_SPAN_CORRUPTION_JSONL}")


if __name__ == "__main__":
    main()