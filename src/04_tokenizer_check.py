import sentencepiece as spm

from config import (
    TOKENIZER_MODEL_PATH,
    VOCAB_SIZE,
    PAD_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,
    USER_DEFINED_SYMBOLS,
)


def check_token_exists(sp: spm.SentencePieceProcessor, token: str) -> None:
    """Print whether a token exists in the tokenizer vocabulary."""
    token_id = sp.piece_to_id(token)

    if token_id == sp.unk_id() and token != UNK_TOKEN:
        print(f"MISSING: {token}")
    else:
        print(f"FOUND: {token} -> id {token_id}")


def main() -> None:
    """Run basic sanity checks on the trained SentencePiece tokenizer."""
    print("Loading tokenizer...")
    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_MODEL_PATH))

    print("\n=== Basic tokenizer info ===")
    print(f"Tokenizer model: {TOKENIZER_MODEL_PATH}")
    print(f"Vocab size: {sp.get_piece_size()}")
    print(f"Expected vocab size: {VOCAB_SIZE}")

    print("\n=== Special token IDs ===")
    print(f"{PAD_TOKEN}: {sp.piece_to_id(PAD_TOKEN)}")
    print(f"{UNK_TOKEN}: {sp.piece_to_id(UNK_TOKEN)}")
    print(f"<s>: {sp.piece_to_id('<s>')}")
    print(f"{EOS_TOKEN}: {sp.piece_to_id(EOS_TOKEN)}")

    print("\n=== Required token checks ===")
    check_token_exists(sp, PAD_TOKEN)
    check_token_exists(sp, UNK_TOKEN)
    check_token_exists(sp, "<s>")
    check_token_exists(sp, EOS_TOKEN)

    print("\n=== Sentinel token checks ===")
    missing_count = 0

    for token in USER_DEFINED_SYMBOLS:
        token_id = sp.piece_to_id(token)

        if token_id == sp.unk_id():
            print(f"MISSING: {token}")
            missing_count += 1

    if missing_count == 0:
        print("All 100 sentinel tokens were found.")
    else:
        print(f"{missing_count} sentinel tokens are missing.")

    print("\n=== Tokenization example ===")
    example = """
    public int add(int a, int b) {
        return a + b;
    }
    """.strip()

    pieces = sp.encode(example, out_type=str)
    ids = sp.encode(example, out_type=int)

    print("Original:")
    print(example)

    print("\nPieces:")
    print(pieces)

    print("\nIDs:")
    print(ids)

    print("\nDecoded:")
    print(sp.decode(ids))

    print("\nTokenizer sanity check complete.")


if __name__ == "__main__":
    main()