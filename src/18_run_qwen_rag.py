import json

import faiss
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import DATA_DIR, PROCESSED_DIR, PREDICTIONS_DIR


TEST_PATH = PROCESSED_DIR / "finetune_test.jsonl"

INDEX_PATH = DATA_DIR / "rag/faiss.index"
META_PATH = DATA_DIR / "rag/meta.json"

OUTPUT_PATH = PREDICTIONS_DIR / "qwen_rag_predictions.jsonl"

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
EMBED_MODEL_NAME = "microsoft/codebert-base"

LIMIT = 200
TOP_K = 3


def load_test_examples(limit=None):
    """Load test examples."""
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


def load_rag_resources():
    """Load FAISS index and metadata."""
    print("Loading FAISS index...")
    index = faiss.read_index(str(INDEX_PATH))

    print("Loading metadata...")
    with open(META_PATH, "r", encoding="utf-8") as file:
        meta = json.load(file)

    return index, meta["buggy"], meta["fixed"]


def retrieve_examples(query, embed_model, index, train_buggy, train_fixed, k=3):
    """Retrieve k similar buggy/fixed training examples."""
    query_embedding = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)

    retrieved = []

    for idx in indices[0]:
        retrieved.append(
            {
                "buggy": train_buggy[idx],
                "fixed": train_fixed[idx],
            }
        )

    return retrieved


def build_rag_prompt(buggy_code, retrieved_examples):
    """Build a 3-shot RAG prompt."""
    examples_text = ""

    for i, example in enumerate(retrieved_examples, start=1):
        examples_text += f"""Example {i}

Buggy code:
{example["buggy"]}

Fixed code:
{example["fixed"]}

"""

    return f"""You are a Java bug fixing assistant.

Use the following buggy-to-fixed examples to guide your fix.

{examples_text}

Now fix the following buggy Java method.

Buggy code:
{buggy_code}

Return only the corrected Java method. Do not explain.
"""


def extract_response(decoded_text, prompt):
    """Remove prompt from decoded output if present."""
    if decoded_text.startswith(prompt):
        return decoded_text[len(prompt):].strip()

    return decoded_text.strip()


def normalize_code(text):
    """Normalize code for exact-match comparison."""
    return " ".join(text.strip().split())


def main():
    """Run Qwen with RAG on bug-fixing examples."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading test examples...")
    examples = load_test_examples(limit=LIMIT)
    print(f"Loaded {len(examples)} examples.")

    index, train_buggy, train_fixed = load_rag_resources()

    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

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
            retrieved = retrieve_examples(
                query=item["buggy"],
                embed_model=embed_model,
                index=index,
                train_buggy=train_buggy,
                train_fixed=train_fixed,
                k=TOP_K,
            )

            prompt = build_rag_prompt(item["buggy"], retrieved)

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=3072,
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
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
                        "retrieved_examples": retrieved,
                        "prediction": prediction,
                        "exact_match": is_exact,
                    }
                )
                + "\n"
            )

    exact_match_accuracy = exact_matches / len(examples)

    print(f"\nQwen RAG exact match: {exact_match_accuracy:.4f}")
    print(f"Saved predictions to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()