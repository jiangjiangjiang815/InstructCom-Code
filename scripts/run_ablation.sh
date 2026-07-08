#!/bin/bash
set -euo pipefail

# Reproduce Table 3 ablations from the repository root:
#   bash scripts/run_ablation.sh
#
# Override any path below before running if your local dataset, split, or
# checkpoint locations differ.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

TRAIN_COT_CONFIG="${TRAIN_COT_CONFIG:-configs/train_cot.json}"
EVAL_COT_CONFIG="${EVAL_COT_CONFIG:-configs/evaluate_multiseed.json}"
EVAL_NOCOT_CONFIG="${EVAL_NOCOT_CONFIG:-configs/evaluate_nocot.json}"

FULL_CHECKPOINT_PATH="${FULL_CHECKPOINT_PATH:-output/synthetic-2000-20/cot_run/final}"
NOCOT_CHECKPOINT_PATH="${NOCOT_CHECKPOINT_PATH:-output/NTU2012/525noCOT/final}"
NOMOD_CHECKPOINT_PATH="${NOMOD_CHECKPOINT_PATH:-output/synthetic-2000-20/nomod/final}"

echo "=== Table 3: Ours - Full InstructCom ==="
# Train the full InstructCom model with CoT rationales, LoRA, community state,
# and modularity-guided candidate filtering.
"$PYTHON_BIN" fintune/code/fintune-COT.py --config "$TRAIN_COT_CONFIG"
"$PYTHON_BIN" fintune/code/evaluate_multiseed.py \
  --config "$EVAL_COT_CONFIG" \
  --checkpoint-path "$FULL_CHECKPOINT_PATH" \
  --prompt-style cot

echo "=== Table 3: w/o. LLM ==="
# Remove the LLM component. This uses only modularity to filter/select nodes.
"$PYTHON_BIN" fintune/code/onlymodule.py

echo "=== Table 3: w/o. LoRA ==="
# Use the untuned backbone model directly for community expansion.
"$PYTHON_BIN" fintune/code/evaluate_nolora.py

echo "=== Table 3: w/o. CoT ==="
# Train without CoT-style rationales, then evaluate with the direct-answer
# prompt that matches fintune-noCOT.py.
"$PYTHON_BIN" fintune/code/fintune-noCOT.py
"$PYTHON_BIN" fintune/code/evaluate_multiseed.py \
  --config "$EVAL_NOCOT_CONFIG" \
  --checkpoint-path "$NOCOT_CHECKPOINT_PATH" \
  --prompt-style direct

echo "=== Table 3: w/o. Com ==="
# Remove encoded community information and feed only the 1-hop hyperedge
# structure as an N-set encoding.
"$PYTHON_BIN" fintune/code/fintune-nocomstate.py
"$PYTHON_BIN" fintune/code/evaluate_nocom.py

echo "=== Table 3: w/o. Mod ==="
# Remove modularity-guided candidate filtering. Train the corresponding model,
# then evaluate with the no-modularity expansion script.
"$PYTHON_BIN" fintune/code/fintune-COT.py \
  --config "$TRAIN_COT_CONFIG" \
  --output-dir "$(dirname "$NOMOD_CHECKPOINT_PATH")"
"$PYTHON_BIN" fintune/code/evaluate_nomod.py

echo "All Table 3 ablation commands finished."
