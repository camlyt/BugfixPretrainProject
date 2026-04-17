from pathlib import Path

import sentencepiece as spm

from config import (
    TOKENIZER_DIR,
    PRETRAIN_METHODS_TXT,
    TOKENIZER_PREFIX,
    VOCAB_SIZE,
    MODEL_TYPE,
    PAD_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,
    USER_DEFINED_SYMBOLS,
)


def main() -> None:
    """Train the SentencePiece tokenizer for the project."""
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    model_prefix = str(TOKENIZER_DIR / TOKENIZER_PREFIX)

    print("Training SentencePiece tokenizer...")
    print(f"Input file: {PRETRAIN_METHODS_TXT}")
    print(f"Model prefix: {model_prefix}")

    spm.SentencePieceTrainer.train(
        input=str(PRETRAIN_METHODS_TXT),
        model_prefix=model_prefix,
        vocab_size=VOCAB_SIZE,
        model_type=MODEL_TYPE,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece=PAD_TOKEN,
        unk_piece=UNK_TOKEN,
        bos_piece="<s>",
        eos_piece=EOS_TOKEN,
        user_defined_symbols=USER_DEFINED_SYMBOLS,
        character_coverage=1.0,
        input_sentence_size=1000000,
        shuffle_input_sentence=True,
    )

    print("Tokenizer training complete.")
    print(f"Model saved to: {model_prefix}.model")
    print(f"Vocab saved to: {model_prefix}.vocab")


if __name__ == "__main__":
    main()