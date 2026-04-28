import json
from typing import Dict, List

import sentencepiece as spm
import torch
from torch.utils.data import Dataset
from transformers import T5Config, T5ForConditionalGeneration, Trainer, TrainingArguments, EarlyStoppingCallback

from config import (
    TOKENIZER_MODEL_PATH,
    FINETUNE_NO_PT_DIR,
    LOGS_DIR,
    PROCESSED_DIR,
)


FINETUNE_TRAIN_JSONL = PROCESSED_DIR / "finetune_train.jsonl"
FINETUNE_VALID_JSONL = PROCESSED_DIR / "finetune_valid.jsonl"
FINETUNE_NO_PT_LOG = LOGS_DIR / "finetune_no_pt_log.json"

MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 512


class BugFixDataset(Dataset):
    """Dataset for bug-fixing fine-tuning."""

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


class DataCollatorForBugFixing:
    """Pad input IDs and labels for T5 bug-fixing training."""

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
            padded = [token_id if token_id != self.pad_token_id else -100 for token_id in padded]
            padded_labels.append(padded)

        return {
            "input_ids": torch.tensor(padded_inputs, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }


def build_fresh_t5_model() -> T5ForConditionalGeneration:
    """Build a fresh T5-small model from scratch using the custom tokenizer."""
    print("Loading SentencePiece tokenizer...")
    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_MODEL_PATH))

    vocab_size = sp.get_piece_size()
    pad_id = sp.piece_to_id("<pad>")
    eos_id = sp.piece_to_id("</s>")
    bos_id = sp.piece_to_id("<s>")

    print("\n=== Fresh model token IDs ===")
    print(f"pad_id: {pad_id}")
    print(f"eos_id: {eos_id}")
    print(f"bos_id: {bos_id}")
    print(f"vocab_size: {vocab_size}")

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


def save_logs(trainer: Trainer) -> None:
    """Save trainer logs."""
    with open(FINETUNE_NO_PT_LOG, "w", encoding="utf-8") as file:
        json.dump(trainer.state.log_history, file, indent=2)

    print(f"Saved fine-tuning logs to: {FINETUNE_NO_PT_LOG}")


def main() -> None:
    """Fine-tune a fresh T5 model on bug fixing without pretraining."""
    print("Building fresh T5-small model from scratch...")
    model = build_fresh_t5_model()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    pad_token_id = model.config.pad_token_id
    print(f"pad_token_id: {pad_token_id}")

    print("\nLoading fine-tuning datasets...")
    train_dataset = BugFixDataset(FINETUNE_TRAIN_JSONL)
    valid_dataset = BugFixDataset(FINETUNE_VALID_JSONL)

    print(f"Train examples: {len(train_dataset)}")
    print(f"Validation examples: {len(valid_dataset)}")

    data_collator = DataCollatorForBugFixing(pad_token_id=pad_token_id)

    training_args = TrainingArguments(
        output_dir=str(FINETUNE_NO_PT_DIR),
        overwrite_output_dir=True,

        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,

        learning_rate=3e-4,
        weight_decay=0.01,
        warmup_steps=500,

        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,

        report_to="none",
        fp16=torch.cuda.is_available(),

        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("Starting fine-tuning WITHOUT pretraining...")
    trainer.train()

    final_dir = FINETUNE_NO_PT_DIR / "final"
    print(f"Saving final model to: {final_dir}")
    trainer.save_model(str(final_dir))

    save_logs(trainer)

    print("Fine-tuning WITHOUT pretraining complete.")


if __name__ == "__main__":
    main()