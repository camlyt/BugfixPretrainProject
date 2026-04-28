import random
from typing import List, Tuple

import sentencepiece as spm

from config import TOKENIZER_MODEL_PATH, SEED


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
    """
    Create corrupted input IDs and target IDs using T5-style sentinel tokens.

    Input example:
        A B C D E F
    If C D and F are masked:
        corrupted input: A B <extra_id_0> E <extra_id_1>
        target: <extra_id_0> C D <extra_id_1> F </s>
    """
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

            # Add sentinel to corrupted input
            corrupted_input.append(sentinel_id)

            # Add sentinel and removed tokens to target
            target.append(sentinel_id)
            for pos in span:
                target.append(token_ids[pos])

            # Skip the full masked span
            i = span[-1] + 1
        else:
            corrupted_input.append(token_ids[i])
            i += 1

    # End target with EOS
    target.append(sp.eos_id())

    return corrupted_input, target


def show_example(sp: spm.SentencePieceProcessor, text: str) -> None:
    """Print original, corrupted input, and target for inspection."""
    token_ids = sp.encode(text, out_type=int)

    corrupted_input, target = apply_span_corruption(sp, token_ids)

    print("\n=== Original text ===")
    print(text)

    print("\n=== Original pieces ===")
    print(sp.id_to_piece(token_ids))

    print("\n=== Corrupted input pieces ===")
    print(sp.id_to_piece(corrupted_input))

    print("\n=== Target pieces ===")
    print(sp.id_to_piece(target))

    print("\n=== Decoded corrupted input ===")
    print(sp.decode(corrupted_input))

    print("\n=== Decoded target ===")
    print(sp.decode(target))

    print("\nOriginal length:", len(token_ids))
    print("Corrupted input length:", len(corrupted_input))
    print("Target length:", len(target))


def main() -> None:
    """Test span corruption on a few Java examples."""
    random.seed(SEED)

    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_MODEL_PATH))

    examples = [
        """
        public int add(int a, int b) {
            return a + b;
        }
        """.strip(),
        """
        public boolean isEven(int number) {
            if (number % 2 == 0) {
                return true;
            }
            return false;
        }
        """.strip(),
        """
        public String greet(String name) {
            String message = "Hello, " + name;
            return message;
        }
        """.strip(),
    ]

    for example in examples:
        show_example(sp, example)
        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()