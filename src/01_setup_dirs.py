from pathlib import Path

from config import (
    RAW_DIR,
    PROCESSED_DIR,
    SAMPLES_DIR,
    TOKENIZER_DIR,
    PRETRAIN_CKPT_DIR,
    FINETUNE_WITH_PT_DIR,
    FINETUNE_NO_PT_DIR,
    LOGS_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
    NOTEBOOKS_DIR,
)


def make_dir(path: Path) -> None:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)
    print(f"Created or already exists: {path}")


def main() -> None:
    """Create the project folder structure."""
    dirs_to_make = [
        RAW_DIR,
        PROCESSED_DIR,
        SAMPLES_DIR,
        TOKENIZER_DIR,
        PRETRAIN_CKPT_DIR,
        FINETUNE_WITH_PT_DIR,
        FINETUNE_NO_PT_DIR,
        LOGS_DIR,
        METRICS_DIR,
        PREDICTIONS_DIR,
        NOTEBOOKS_DIR,
    ]

    for directory in dirs_to_make:
        make_dir(directory)

    print("\nProject folders are ready.")


if __name__ == "__main__":
    main()