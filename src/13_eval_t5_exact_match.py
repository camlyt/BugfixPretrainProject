import json
from pathlib import Path
from typing import Dict, List

import sentencepiece as spm
import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration

from config import (
    TOKENIZER_MODEL_PATH,
    FINETUNE_WITH_PT_DIR,
    FINETUNE_NO_PT_DIR,
    PROCESSED_DIR,
    PREDICTIONS_DIR,
    METRICS_DIR,
)


FINETUNE_TEST_JSONL = PROCESSED_DIR / "finetune_test.jsonl"

WITH_PT_MODEL_DIR = FINETUNE_WITH_PT_DIR / "final"
NO_PT_MODEL_DIR = FINETUNE_NO_PT_DIR / "final"

MAX_INPUT_LENGTH = 512

class SimpleTokenizerWrapper:
    """Small wrapper around SentencePiece for encode/decode."""

    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))

    def encode(self, text: str) -> List[int]:
        return self.sp.encode(text, out_type=int)[:MAX_INPUT_LENGTH]

    def decode(self, ids: List[int]) -> str:
        return self.sp.decode(ids)

    @property
    def pad_id(self) -> int:
        return self.sp.piece_to_id("<pad>")

    @property
    def eos_id(self) -> int:
        return self.sp.piece_to_id("</s>")


def load_test_examples(limit=None) -> List[Dict[str, str]]:
    """Load test examples."""
    examples = []

    with open(FINETUNE_TEST_JSONL, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file):
            if limit is not None and line_number >= limit:
                break

            item = json.loads(line)
            examples.append(
                {
                    "buggy": item["buggy"],
                    "fixed": item["fixed"],
                }
            )

    return examples


def normalize_code(text: str) -> str:
    """Normalize code for exact-match comparison."""
    return " ".join(text.strip().split())


def evaluate_model(
    model_dir: Path,
    model_name: str,
    tokenizer: SimpleTokenizerWrapper,
    examples: List[Dict[str, str]],
) -> Dict[str, float]:
    """Generate fixes and compute exact match."""
    print(f"\nLoading model: {model_name}")
    print(f"Path: {model_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = T5ForConditionalGeneration.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    predictions_path = PREDICTIONS_DIR / f"{model_name}_predictions.jsonl"

    exact_matches = 0

    with open(predictions_path, "w", encoding="utf-8") as out_file:
        for example in tqdm(examples):
            input_ids = tokenizer.encode(example["buggy"])

            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
            attention_mask = torch.ones_like(input_tensor).to(device)

            with torch.no_grad():
                generated = model.generate(
                    input_ids=input_tensor,
                    attention_mask=attention_mask,
                    max_new_tokens=128,
                    num_beams=2,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.2,
                    early_stopping=True,
                    pad_token_id=tokenizer.pad_id,
                    eos_token_id=tokenizer.eos_id,
                )

            prediction = tokenizer.decode(generated[0].tolist())
            reference = example["fixed"]

            is_exact = normalize_code(prediction) == normalize_code(reference)

            if is_exact:
                exact_matches += 1

            out_file.write(
                json.dumps(
                    {
                        "buggy": example["buggy"],
                        "reference": reference,
                        "prediction": prediction,
                        "exact_match": is_exact,
                    }
                )
                + "\n"
            )

    exact_match_score = exact_matches / len(examples)

    metrics = {
        "model": model_name,
        "num_examples": len(examples),
        "exact_matches": exact_matches,
        "exact_match_accuracy": exact_match_score,
    }

    metrics_path = METRICS_DIR / f"{model_name}_exact_match.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"\n{model_name} exact match: {exact_match_score:.4f}")
    print(f"Saved predictions to: {predictions_path}")
    print(f"Saved metrics to: {metrics_path}")

    return metrics


def main() -> None:
    """Evaluate both T5 models using exact match."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = SimpleTokenizerWrapper(TOKENIZER_MODEL_PATH)

    print("Loading test set...")
    examples = load_test_examples()
    print(f"Loaded {len(examples)} test examples.")

    results = []

    results.append(
        evaluate_model(
            WITH_PT_MODEL_DIR,
            "t5_with_pretraining",
            tokenizer,
            examples,
        )
    )

    results.append(
        evaluate_model(
            NO_PT_MODEL_DIR,
            "t5_no_pretraining",
            tokenizer,
            examples,
        )
    )

    combined_path = METRICS_DIR / "t5_exact_match_summary.json"

    with open(combined_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(f"\nSaved combined summary to: {combined_path}")


if __name__ == "__main__":
    main()