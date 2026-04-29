import json
import os
import sys

from config import PREDICTIONS_DIR, METRICS_DIR

# Add CodeBLEU to path
sys.path.append("/content/CodeBLEU")

from calc_code_bleu import calc_code_bleu


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

    score = calc_code_bleu(
        refs,
        preds,
        lang=LANG,
    )

    result = {
        "model": model_name,
        "codebleu": score,
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