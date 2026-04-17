from pathlib import Path

# Root of the repository
ROOT_DIR = Path(__file__).resolve().parent

# Main directories
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"

TOKENIZER_DIR = ROOT_DIR / "tokenizer"

CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
PRETRAIN_CKPT_DIR = CHECKPOINTS_DIR / "pretrain"
FINETUNE_WITH_PT_DIR = CHECKPOINTS_DIR / "finetune_with_pt"
FINETUNE_NO_PT_DIR = CHECKPOINTS_DIR / "finetune_no_pt"

OUTPUTS_DIR = ROOT_DIR / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
SRC_DIR = ROOT_DIR / "src"

# Reproducibility
SEED = 42

# Pretraining corpus settings
PRETRAIN_SAMPLE_SIZE = 50000
MIN_TOKENS = 10
MAX_TOKENS = 512

# Tokenizer settings
TOKENIZER_PREFIX = "spm_unigram_16k"
VOCAB_SIZE = 16384
MODEL_TYPE = "unigram"

# T5 special tokens required by the assignment
USER_DEFINED_SYMBOLS = [f"<extra_id_{i}>" for i in range(100)]

PAD_TOKEN = "<pad>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<unk>"

# Data file paths
PRETRAIN_METHODS_TXT = PROCESSED_DIR / "pretrain_methods_50k.txt"
PRETRAIN_METHODS_JSONL = PROCESSED_DIR / "pretrain_methods_50k.jsonl"

# Tokenizer output paths
TOKENIZER_MODEL_PATH = TOKENIZER_DIR / f"{TOKENIZER_PREFIX}.model"
TOKENIZER_VOCAB_PATH = TOKENIZER_DIR / f"{TOKENIZER_PREFIX}.vocab"