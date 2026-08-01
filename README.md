# InstructCom-Code

InstructCom is an Instruction-tuned large language model (LLM) for semi-supervised hypergraph local Community detection. The full project workflow is:
1. Run `generate_data/main.py` to generate an instruction-tuning dataset from a hypergraph dataset. During this step, replace the API key with your own key and set the required input/output parameters.
2. Feed the generated instruction-tuning dataset into `code/train/main.py` to perform QLoRA / LoRA fine-tuning.
3. Record the trained LoRA adapter path.
4. Use the base model path and the trained LoRA adapter path as inputs to `code/inference/InstrucCom.py` for model inference and community detection evaluation.
5. Read the output JSON file to inspect the predicted communities and evaluation metrics.

> Note: In the current repository, the training entrypoint is `code/train/main.py`, and the inference entrypoint is `code/inference/InstrucCom.py`.

Each dataset directory is expected to contain:

- `labels.txt`: one node label per line. The line number, starting from 1, is used as the node ID.
- `hyperedges.txt`: one hyperedge per line. Node IDs are separated by commas, for example `1,3,7,9`.

## Environment setup

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Step 1: Instruction sample construction
This script reads `labels.txt` and `hyperedges.txt`, builds ground-truth communities from labels, splits communities into train/test sets, constructs community-expansion prompts, calls the Qwen API to generate reasoning text, and saves the resulting instruction-tuning samples in JSONL format.

### Configure your API key

Before running the script, replace the API key with your own key. The current code uses `api_key` inside `call_qwen_api()`:

### Example

Run from the project root:

```bash
python generate_data/main.py \
  --base-path datasets/contact-primary-school \
  --output-dir datasets/contact-primary-school/generated \
  --data-output train_expand.jsonl \
  --stop-output train_stop.jsonl \
  --candidate-top-k 6
```
### Generated files

The example above writes:

```text
datasets/contact-primary-school/generated/train_expand.jsonl
datasets/contact-primary-school/generated/train_stop.jsonl
datasets/contact-primary-school/generated/contact-primary-school_community_split.json
```

The files mean:

- `train_expand.jsonl`: expansion samples where the model learns to select 1-4 candidate nodes.
- `train_stop.jsonl`: STOP samples where the model learns when to stop expanding.
- `contact-primary-school_community_split.json`: train/test community label split. This file is needed during inference through `--split-file-path`.

The training script accepts one `--data-path`. If you want to use both expansion and STOP samples, merge them first. 

## Step 2: Fine-tuning
This script reads the JSONL instruction-tuning data, fine-tunes a Qwen-style causal language model with 4-bit QLoRA, and saves the final LoRA adapter under the output directory.

### Example

```bash
python code/train/main.py \
  --model-name code/qwen25-14b \
  --data-path datasets/contact-primary-school/generated/train_all.jsonl \
  --output-dir output/contact-primary-school/cot_lora \
  --max-length 4096 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --learning-rate 5e-5 \
  --num-train-epochs 5 \
  --save-steps 50 \
  --lora-r 128 \
  --lora-alpha 32 \
  --lora-dropout 0.05
```
### Training output

After training, the final LoRA adapter is saved to:

```text
output/contact-primary-school/cot_lora/final
```

Record this path. It is used as `--checkpoint-path` during inference.

## Step 3: Inference for local community expansion

Entrypoint:

```text
code/inference/InstrucCom.py
```
The inference script loads the base model and the trained LoRA adapter, then performs community expansion on the test communities.
### Example

```bash
python code/inference/InstrucCom.py \
  --model-name code/qwen25-14b \
  --checkpoint-path output/contact-primary-school/cot_lora/final \
  --base-path datasets/contact-primary-school \
  --label-file labels.txt \
  --edge-file hyperedges.txt \
  --split-file-path datasets/contact-primary-school/generated/contact-primary-school_community_split.json \
  --output-path output/contact-primary-school/eval_results.json \
  --log-file output/contact-primary-school/debug_expansion.log \
  --max-candidates 6 \
  --max-new-tokens 1024 \
  --num-seeds-per-comm-test 100 \
  --prompt-style cot
```

This produces:

```text
output/contact-primary-school/eval_results.json
output/contact-primary-school/debug_expansion.log
```

### `summary`

`summary` reports average metrics over all tested communities and all seed runs:

- `jaccard`: Jaccard similarity between predicted and ground-truth communities. Higher is better.
- `f1`: harmonic mean of precision and recall. Higher is better.
- `precision`: fraction of predicted nodes that are actually in the ground-truth community.
- `recall`: fraction of ground-truth nodes recovered by the prediction.
