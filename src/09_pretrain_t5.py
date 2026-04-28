import json
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import T5Config, T5ForConditionalGeneration, Trainer, TrainingArguments
import sentencepiece as spm

from config import (
    TOKENIZER_MODEL_PATH,
    PROCESSED_DIR,
    PRETRAIN_CKPT_DIR,
    LOGS_DIR,
)


PRETRAIN_SPAN_CORRUPTION_JSONL = PROCESSED_DIR / "pretrain_span_corruption.jsonl"
PRETRAIN_LOSS_LOG = LOGS_DIR / "pretrain_loss_log.json"

MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 256


class SpanCorruptionDataset(Dataset):
    """PyTorch dataset for T5 span-corruption pretraining."""

    def __init__(self, jsonl_path):
        self.examples = []

        with open(jsonl_path, "r", encoding="utf-8") as file:
            for line in file:
                item = json.loads(line)
                self.examples.append(
                    {
                        "input_ids": item["input_ids"][:MAX_INPUT_LENGTH],
                        "labels": item["labels"][:MAX_TARGET_LENGTH],
                    }
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class DataCollatorForT5Pretraining:
    """Pads input_ids and labels for T5 training."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        input_ids = [item["input_ids"] for item in batch]
        labels = [item["labels"] for item in batch]

        max_input_len = max(len(ids) for ids in input_ids)
        max_label_len = max(len(ids) for ids in labels)

        padded_inputs = []
        attention_masks = []
        padded_labels = []

        for ids in input_ids:
            padding_len = max_input_len - len(ids)
            padded_inputs.append(ids + [self.pad_token_id] * padding_len)
            attention_masks.append([1] * len(ids) + [0] * padding_len)

        for label_ids in labels:
            padding_len = max_label_len - len(label_ids)
            padded = label_ids + [self.pad_token_id] * padding_len

            # T5 ignores label positions with -100
            padded = [token_id if token_id != self.pad_token_id else -100 for token_id in padded]
            padded_labels.append(padded)

        return {
            "input_ids": torch.tensor(padded_inputs, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }


def build_t5_model(sp: spm.SentencePieceProcessor) -> T5ForConditionalGeneration:
    """Build a T5-small model from scratch using the custom tokenizer size."""
    vocab_size = sp.get_piece_size()

    pad_id = sp.piece_to_id("<pad>")
    eos_id = sp.piece_to_id("</s>")
    bos_id = sp.piece_to_id("<s>")

    config = T5Config(
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

    model = T5ForConditionalGeneration(config=config)
    model.resize_token_embeddings(vocab_size)

    return model


def save_epoch_losses(trainer: Trainer) -> None:
    """Save training loss logs from the Trainer."""
    logs = trainer.state.log_history

    with open(PRETRAIN_LOSS_LOG, "w", encoding="utf-8") as file:
        json.dump(logs, file, indent=2)

    print(f"Saved training logs to: {PRETRAIN_LOSS_LOG}")


def main() -> None:
    """Pretrain T5-small using span corruption for 3 epochs."""
    print("Loading SentencePiece tokenizer...")
    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_MODEL_PATH))

    pad_id = sp.piece_to_id("<pad>")

    print("Loading pretraining dataset...")
    train_dataset = SpanCorruptionDataset(PRETRAIN_SPAN_CORRUPTION_JSONL)
    print(f"Loaded {len(train_dataset)} examples.")

    print("Building T5-small model from scratch...")
    model = build_t5_model(sp)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    data_collator = DataCollatorForT5Pretraining(pad_token_id=pad_id)

    training_args = TrainingArguments(
        output_dir=str(PRETRAIN_CKPT_DIR),
        overwrite_output_dir=True,

        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,

        learning_rate=5e-4,
        weight_decay=0.01,
        warmup_steps=500,

        logging_dir=str(LOGS_DIR),
        logging_steps=100,
        save_strategy="epoch",

        report_to="none",
        fp16=torch.cuda.is_available(),

        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print("Starting pretraining...")
    trainer.train()

    print("Saving final pre-trained model...")
    trainer.save_model(str(PRETRAIN_CKPT_DIR / "final"))

    save_epoch_losses(trainer)

    print(f"Final pre-trained model saved to: {PRETRAIN_CKPT_DIR / 'final'}")


if __name__ == "__main__":
    main()