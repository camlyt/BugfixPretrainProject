import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import DATA_DIR


INDEX_PATH = DATA_DIR / "rag/faiss.index"
META_PATH = DATA_DIR / "rag/meta.json"


def load_index():
    print("Loading FAISS index...")
    index = faiss.read_index(str(INDEX_PATH))
    return index


def load_meta():
    print("Loading metadata...")
    with open(META_PATH, "r") as f:
        data = json.load(f)
    return data["buggy"], data["fixed"]


def retrieve(query, model, index, buggy, fixed, k=3):
    q_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, k)

    results = []
    for idx in I[0]:
        results.append((buggy[idx], fixed[idx]))

    return results


def main():
    model = SentenceTransformer("microsoft/codebert-base")

    index = load_index()
    buggy, fixed = load_meta()

    # Test query (use real example)
    test_query = """
public int add(int a, int b) {
    return a - b;
}
"""

    print("\nQuery:")
    print(test_query)

    results = retrieve(test_query, model, index, buggy, fixed, k=3)

    print("\nTop 3 retrieved examples:\n")

    for i, (b, f) in enumerate(results):
        print(f"--- Example {i+1} ---")
        print("Buggy:\n", b)
        print("Fixed:\n", f)
        print()


if __name__ == "__main__":
    main()