from transformers import T5Config, T5ForConditionalGeneration
import sentencepiece as spm

from config import (
    TOKENIZER_MODEL_PATH,
)


def main() -> None:
    print("Loading tokenizer...")
    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_MODEL_PATH))

    vocab_size = sp.get_piece_size()

    pad_id = sp.piece_to_id("<pad>")
    eos_id = sp.piece_to_id("</s>")
    bos_id = sp.piece_to_id("<s>")

    print("\n=== Token IDs ===")
    print(f"pad_id: {pad_id}")
    print(f"eos_id: {eos_id}")
    print(f"bos_id: {bos_id}")
    print(f"vocab_size: {vocab_size}")

    print("\nBuilding T5-small config...")

    t5_config = T5Config(
        vocab_size=vocab_size,
        d_model=512,
        d_ff=2048,
        d_kv=64,
        num_heads=8,
        num_layers=6,
        num_decoder_layers=6,
        decoder_start_token_id=pad_id,
        eos_token_id=eos_id,
        bos_token_id=bos_id,
        pad_token_id=pad_id,
    )

    model = T5ForConditionalGeneration(config=t5_config)

    print("\nResizing embeddings to tokenizer vocab...")
    model.resize_token_embeddings(vocab_size)

    total_params = sum(p.numel() for p in model.parameters())

    print("\n=== Model Summary ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Expected ~60M parameters")

    print("\nModel successfully built from scratch.")


if __name__ == "__main__":
    main()