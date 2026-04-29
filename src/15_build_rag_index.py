import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from config import DATA_DIR


TRAIN_PATH = DATA_DIR / "processed/finetune_train.jsonl"
INDEX_PATH = DATA_DIR / "rag/faiss.index"
META_PATH = DATA_DIR / "rag/meta.json"


def load_data(path):
    buggy = []
    fixed = []

    with open(path, "r") as f:
        for line in f:
            item = json.loads(line)
            buggy.append(item["buggy"])
            fixed.append(item["fixed"])

    return buggy, fixed


def main():
    print("Loading training data...")
    buggy, fixed = load_data(TRAIN_PATH)

    print(f"Loaded {len(buggy)} examples")

    print("Loading embedding model...")
    model = SentenceTransformer("microsoft/codebert-base")

    print("Encoding buggy code...")
    embeddings = model.encode(
        buggy,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=64
    )

    print("Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    print("Saving index...")
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))

    print("Saving metadata...")
    with open(META_PATH, "w") as f:
        json.dump({
            "buggy": buggy,
            "fixed": fixed
        }, f)

    print("Done.")


if __name__ == "__main__":
    main()