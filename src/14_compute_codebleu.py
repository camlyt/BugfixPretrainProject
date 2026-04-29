import json
import os

from codebleu import calc_codebleu

from config import PREDICTIONS_DIR, METRICS_DIR


PRED_FILES = {
    "t5_with_pretraining": PREDICTIONS_DIR / "t5_with_pretraining_predictions.jsonl",
    "t5_no_pretraining": PREDICTIONS_DIR / "t5_no_pretraining_predictions.jsonl",
}


LANG = "java"


def load_predictions(path):
    """Load reference and prediction strings from a JSONL file."""
    references = []
    predictions = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            references.append(item["reference"])
            predictions.append(item["prediction"])

    return references, predictions


def evaluate(model_name, path):
    """Compute CodeBLEU for one model."""
    print(f"\nEvaluating {model_name}...")
    print(f"Prediction file: {path}")

    references, predictions = load_predictions(path)

    result = calc_codebleu(
        references,
        predictions,
        lang=LANG,
    )

    output = {
        "model": model_name,
        "num_examples": len(predictions),
        "codebleu": result["codebleu"],
        "details": result,
    }

    output_path = METRICS_DIR / f"{model_name}_codebleu.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print(f"{model_name} CodeBLEU: {result['codebleu']:.4f}")
    print(f"Saved to: {output_path}")

    return output


def main():
    """Compute CodeBLEU for both T5 models."""
    os.makedirs(METRICS_DIR, exist_ok=True)

    results = []

    for name, path in PRED_FILES.items():
        results.append(evaluate(name, path))

    summary_path = METRICS_DIR / "codebleu_summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()