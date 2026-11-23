# Personal AI — Transformer trained from random initialization

A complete decoder-only Transformer language-model project built with PyTorch.
It uses **no pretrained weights, hosted model, inference API, or external AI service**.
All parameters are initialized locally and learned by next-token prediction with
cross-entropy loss, backpropagation, AdamW updates, dropout, weight decay, validation,
early stopping, and a final held-out test evaluation.

## Important, honest limitation

A model trained from random initialization only becomes useful to the extent that it
receives enough high-quality text and compute. This repository does not pretend that
a small demo corpus creates a GPT-level assistant. The data pipeline downloads public
domain books from Project Gutenberg (plain HTTPS files, not an API), removes boilerplate,
and creates train/validation/test splits. You can also place your own licensed `.txt`
files in `data/raw/`. The downloaded corpus is real training data, not code-embedded
answers. Start with the small default configuration to verify the pipeline; use the
larger configuration and a substantial corpus for serious experiments.

## Requirements

- Python 3.10+
- PyTorch 2.2+ (CPU works; CUDA is strongly recommended)
- `pip install -r requirements.txt`

No Hugging Face package, pretrained model, API key, or external inference service is used.

## Quick start

```bash
# 1. Install dependencies
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt

# 2. Download public-domain training texts into data/raw/
python scripts/download_corpus.py

# 3. Build vocabulary and non-overlapping document-level splits
python scripts/prepare_data.py

# 4. Train from random initialization
python train.py --config configs/small.json

# 5. Evaluate the best validation checkpoint on the untouched test split
python evaluate.py --checkpoint runs/best.pt

# 6. Generate from a prompt not used as a training sample
python chat.py --checkpoint runs/best.pt
```

The downloader is optional. You may use your own `.txt` corpus. Respect the license
and terms for any texts you add. The default downloader uses public-domain Project
Gutenberg works; check local legal status before redistributing their text.

## Data split and leakage controls

- Raw source files are kept separate from Python code in `data/raw/`.
- Splitting is by source document, not random chunks, so a document cannot leak across
  train/validation/test.
- `data/processed/manifest.json` records which source document belongs to each split.
- The test split is not used for gradient updates, checkpoint selection, or early stopping.
- The tokenizer vocabulary is built from training documents only.
- Test loss is computed once after training by `evaluate.py`.

## Training controls

- Decoder-only causal Transformer, learned positional embeddings, pre-layer norm,
  multi-head causal self-attention, GELU MLP, residual dropout.
- AdamW with decoupled weight decay, gradient clipping, mixed precision on CUDA,
  periodic validation, best-checkpoint selection, and patience-based early stopping.
- The training log records train and validation loss. `evaluate.py` reports final test
  loss and perplexity.
- The chat prompt is supplied at inference time; generated output is not hardcoded.

## Configuration

Edit `configs/small.json` for a low-resource smoke run. `configs/medium.json` is a
more substantial starting point. Increase corpus size, context length, model width,
layers, steps, and batch size only when memory/compute allow.

## Outputs

- `runs/best.pt`: best validation checkpoint
- `runs/last.pt`: latest checkpoint
- `runs/history.jsonl`: per-evaluation train/validation metrics
- `runs/test_metrics.json`: final held-out test metrics
- `runs/generated_unseen_prompt.txt`: example inference output

A generated response may be incoherent if the model is undertrained or the corpus is
too small. The project never fabricates successful metrics or claims that training
has happened when it has not.
