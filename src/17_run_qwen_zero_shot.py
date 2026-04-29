import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import PROCESSED_DIR, PREDICTIONS_DIR


TEST_PATH = PROCESSED_DIR / "finetune_test.jsonl"
OUTPUT_PATH = PREDICTIONS_DIR / "qwen_zero_shot_predictions.jsonl"

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
LIMIT = 200


def load_test_examples(limit=None):
    """Load test examples from the fine-tuning test file."""
    examples = []

    with open(TEST_PATH, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file):
            if limit is not None and line_number >= limit:
                break

            item = json.loads(line)
            examples.append(
                {
                    "buggy": item["buggy"],
                    "reference": item["fixed"],
                }
            )

    return examples


def build_prompt(buggy_code):
    """Build zero-shot prompt for Qwen."""
    return f"""You are a Java bug fixing assistant.

Fix the following buggy Java method.

Buggy code:
{buggy_code}

Return only the corrected Java method. Do not explain.
"""


def extract_response(decoded_text, prompt):
    """Remove the prompt from the decoded output if present."""
    if decoded_text.startswith(prompt):
        return decoded_text[len(prompt):].strip()

    return decoded_text.strip()


def normalize_code(text):
    """Normalize code for exact-match comparison."""
    return " ".join(text.strip().split())


def main():
    """Run Qwen zero-shot bug fixing."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading test examples...")
    examples = load_test_examples(limit=LIMIT)
    print(f"Loaded {len(examples)} examples.")

    print("Loading Qwen model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    model.eval()

    exact_matches = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_file:
        for item in tqdm(examples):
            prompt = build_prompt(item["buggy"])

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            prediction = extract_response(decoded, prompt)

            is_exact = normalize_code(prediction) == normalize_code(item["reference"])

            if is_exact:
                exact_matches += 1

            out_file.write(
                json.dumps(
                    {
                        "buggy": item["buggy"],
                        "reference": item["reference"],
                        "prediction": prediction,
                        "exact_match": is_exact,
                    }
                )
                + "\n"
            )

    exact_match_accuracy = exact_matches / len(examples)

    print(f"\nQwen zero-shot exact match: {exact_match_accuracy:.4f}")
    print(f"Saved predictions to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()