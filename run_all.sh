#!/usr/bin/env bash
set -e
python scripts/download_corpus.py
python scripts/prepare_data.py
python train.py --config configs/small.json
python evaluate.py --checkpoint runs/best.pt
python test_unseen_prompt.py
