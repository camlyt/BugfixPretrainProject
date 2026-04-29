# BugfixPretrainProject
## Overview

This project explores automated Java code repair using three approaches:

1. Fine-tuned T5 (with and without pre-training)
2. Zero-shot prompting with Qwen
3. Retrieval-Augmented Generation (RAG) with Qwen

The pipeline is implemented as a sequence of scripts in the src/ directory. Outputs are written to structured folders under data/, checkpoints/, and outputs/.

---

## Installation

Clone the repository and install dependencies:

```
git clone https://github.com/YOUR_USERNAME/BugfixPretrainProject.git
cd BugfixPretrainProject
pip install -r requirements.txt
```

For CodeBLEU evaluation, install compatible versions:

```
pip uninstall -y tree-sitter tree-sitter-java codebleu
pip install tree-sitter==0.22.3 tree-sitter-java==0.21.0 codebleu==0.7.0
```

## Reproducing Results
1. Setup directories
```
PYTHONPATH=. python src/01_setup_dirs.py
```
2. Train tokenizer (only if not already saved)
```
PYTHONPATH=. python src/02_download_pretrain_data.py
PYTHONPATH=. python src/05_filter_pretrain_data.py
PYTHONPATH=. python src/03_train_tokenizer.py
```
3. Prepare fine-tuning dataset
```
PYTHONPATH=. python src/10_prepare_finetune_dataset.py
```
4. Train and evaluate T5 pipelines
```
PYTHONPATH=. python src/09_pretrain_t5.py
PYTHONPATH=. python src/11_finetune_no_pretrain.py
PYTHONPATH=. python src/12_finetune_with_pretrain.py
PYTHONPATH=. python src/13_eval_t5_exact_match.py
PYTHONPATH=. python src/14_compute_codebleu.py
```
5. Build RAG index
```
PYTHONPATH=. python src/15_build_rag_index.py
```
6. Run Qwen pipelines

Zero-shot baseline:
```
PYTHONPATH=. python src/17_run_qwen_zero_shot.py
```
RAG-enhanced:
```
PYTHONPATH=. python src/18_run_qwen_rag.py
```
7. Evaluate Qwen outputs
```
PYTHONPATH=. python src/19_compute_qwen_codebleu.py
```
## Outputs
Model checkpoints
```
checkpoints/pretrain/
checkpoints/finetune_no_pt/
checkpoints/finetune_with_pt/
```
Processed datasets
```
data/processed/
  finetune_train.jsonl
  finetune_valid.jsonl
  finetune_test.jsonl
```
RAG artifacts
```
data/rag/
  faiss.index
  meta.json
  ```
Predictions
```
outputs/predictions/
  t5_with_pretraining_predictions.jsonl
  t5_no_pretraining_predictions.jsonl
  qwen_zero_shot_predictions.jsonl
  qwen_rag_predictions.jsonl
  ```
Metrics
```
outputs/metrics/
  t5_with_pretraining_codebleu.json
  t5_no_pretraining_codebleu.json
  qwen_zero_shot_codebleu.json
  qwen_rag_codebleu.json
  *_summary.json
  ```
Logs
```
outputs/logs/
  pretrain_loss_log.json
  finetune_with_pt_log.json
  finetune_no_pt_log.json
  ```
## Notes
GPU is recommended for training and Qwen inference steps.
Colab was used for compute-heavy stages.
Intermediate outputs (checkpoints, predictions, metrics) can be saved and reused to avoid recomputation.