import json
import os
import sys

from config import PREDICTIONS_DIR, METRICS_DIR

# Add CodeBLEU to path
sys.path.append("/content/CodeBLEU")

from codebleu import calc_codebleu


PRED_FILES = {
    "t5_with_pretraining": PREDICTIONS_DIR / "t5_with_pretraining_predictions.jsonl",
    "t5_no_pretraining": PREDICTIONS_DIR / "t5_no_pretraining_predictions.jsonl",
}


LANG = "java"


def load_predictions(path):
    refs = []
    preds = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            refs.append(item["reference"])
            preds.append(item["prediction"])

    return refs, preds


def evaluate(model_name, path):
    print(f"\nEvaluating {model_name}...")

    refs, preds = load_predictions(path)

    # CodeBLEU expects list of lists for references
    refs = [[r] for r in refs]

    result = calc_codebleu(
        preds,
        refs,
        lang="java"
    )

    score = result["codebleu"]

    output = {
        "model": model_name,
        "codebleu": score,
        "details": result
    }

    output_path = METRICS_DIR / f"{model_name}_codebleu.json"

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"{model_name} CodeBLEU: {score:.4f}")
    return result


def main():
    os.makedirs(METRICS_DIR, exist_ok=True)

    results = []

    for name, path in PRED_FILES.items():
        results.append(evaluate(name, path))

    summary_path = METRICS_DIR / "codebleu_summary.json"

    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()