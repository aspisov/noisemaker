#!/usr/bin/env bash
set -euo pipefail

HYDRA_FULL_ERROR=1 uv run python src/train.py \
  trainer=mps \
  data.batch_size=128 \
  data.num_workers=10
