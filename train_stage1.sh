#!/usr/bin/env bash
set -euo pipefail

# Stage-1 Graph Q-Former training launcher
# Optional overrides:
#   OMNIGRAPH_HF_CACHE=/path/to/cache
#   OMNIGRAPH_HF_ENDPOINT=https://hf-mirror.com

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKDIR"

PYTHON_BIN=${PYTHON_BIN:-python}

$PYTHON_BIN omnigraph/train/train_graph_qfromer.py \
	--batch_size 32 \
	--lr 1e-4 \
	--epochs 20 \
	--dataset_paths data/train_instruct_graphmatch.json data/arxiv_pub_node_st_cot_link_mix.json \
	--text_model_name sentence-transformers/all-MiniLM-L6-v2
