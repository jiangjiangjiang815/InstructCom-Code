"""QLoRA fine-tuning entrypoint with chain-of-thought supervision.

The script keeps the original training recipe, but exposes all experiment
settings through a JSON config file and command-line overrides so runs are
reproducible and easy to document.
"""

import argparse
import json
import logging
import os
from copy import deepcopy

try:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import Dataset as TorchDataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )
    DEPENDENCY_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    torch = None
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None
    DataCollatorForSeq2Seq = None
    Trainer = None
    TrainingArguments = None
    TorchDataset = object
    DEPENDENCY_IMPORT_ERROR = exc


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_CONFIG = {
    "model_name": "fintune/qwen25-14b",
    "data_path": "dataset/synthetic-2000-20/train_cot.jsonl",
    "output_dir": "output/synthetic-2000-20/cot_run",
    "max_length": 4096,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 5e-5,
    "num_train_epochs": 5,
    "logging_steps": 5,
    "save_steps": 50,
    "save_total_limit": 2,
    "optim": "paged_adamw_8bit",
    "bf16": True,
    "max_grad_norm": 1.0,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "linear",
    "gradient_checkpointing": False,
    "dataloader_num_workers": 8,
    "lora_r": 128,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


class InstructionDataset(TorchDataset):
    """JSONL dataset for instruction tuning.

    Each line must contain instruction, input, and output fields. The prompt
    tokens are masked with -100 so the loss is computed only on the answer.
    """

    def __init__(self, json_path, tokenizer, max_length=4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        logger.info("Loading JSONL data from %s...", json_path)
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))
        logger.info("Loaded %d examples.", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = item.get("instruction", "")
        input_dict = item.get("input", {})
        current_community = input_dict.get("current_community", "[]")
        community_stats = input_dict.get("community_state", "[]")
        candidates = input_dict.get("candidates", "[]")
        output = str(item.get("output", "")).strip()

        user_content = (
            f"{instruction}\n\n"
            f"### Current Community Nodes:\n{current_community}\n\n"
            f"### Current Community state:\n{community_stats}\n\n"
            f"### Candidate Neighbors:\n{candidates}\n\n"
            f"Please think step by step and select the nodes."
        )

        messages = [
            {"role": "system", "content": "You are a Hypergraph Community Detection Expert."},
            {"role": "user", "content": user_content},
        ]
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_text = prompt_text + output + self.tokenizer.eos_token

        tokenized_full = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            add_special_tokens=False,
        )
        tokenized_prompt = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )

        input_ids = tokenized_full["input_ids"]
        attention_mask = tokenized_full["attention_mask"]
        prompt_len = len(tokenized_prompt["input_ids"])

        labels = [-100] * len(input_ids)
        for i in range(prompt_len, len(input_ids)):
            labels[i] = input_ids[i]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def ensure_dependencies():
    if DEPENDENCY_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Missing training dependencies. Install them with `pip install -r requirements.txt` "
            "and ensure the correct CUDA-enabled PyTorch build is available."
        ) from DEPENDENCY_IMPORT_ERROR


def load_config(config_path):
    config = deepcopy(DEFAULT_CONFIG)
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        unknown_keys = sorted(set(loaded) - set(DEFAULT_CONFIG))
        if unknown_keys:
            raise ValueError(f"Unknown config keys in {config_path}: {unknown_keys}")
        config.update(loaded)
    return config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Fine-tune a Qwen-style causal LM with QLoRA.")
    parser.add_argument("--config", default=None, help="Path to a JSON experiment config.")
    parser.add_argument("--model-name", dest="model_name", default=None)
    parser.add_argument("--data-path", dest="data_path", default=None)
    parser.add_argument("--output-dir", dest="output_dir", default=None)
    parser.add_argument("--max-length", dest="max_length", type=int, default=None)
    parser.add_argument("--per-device-train-batch-size", dest="per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", dest="gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=None)
    parser.add_argument("--num-train-epochs", dest="num_train_epochs", type=float, default=None)
    parser.add_argument("--logging-steps", dest="logging_steps", type=int, default=None)
    parser.add_argument("--save-steps", dest="save_steps", type=int, default=None)
    parser.add_argument("--save-total-limit", dest="save_total_limit", type=int, default=None)
    parser.add_argument("--optim", default=None)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-grad-norm", dest="max_grad_norm", type=float, default=None)
    parser.add_argument("--warmup-ratio", dest="warmup_ratio", type=float, default=None)
    parser.add_argument("--lr-scheduler-type", dest="lr_scheduler_type", default=None)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dataloader-num-workers", dest="dataloader_num_workers", type=int, default=None)
    parser.add_argument("--lora-r", dest="lora_r", type=int, default=None)
    parser.add_argument("--lora-alpha", dest="lora_alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", dest="lora_dropout", type=float, default=None)
    parser.add_argument(
        "--target-modules",
        dest="target_modules",
        default=None,
        help="Comma-separated LoRA target module names.",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    for key, value in vars(args).items():
        if key == "config" or value is None:
            continue
        config[key] = value

    if isinstance(config["target_modules"], str):
        config["target_modules"] = [item.strip() for item in config["target_modules"].split(",") if item.strip()]

    return config


def create_qlora_model(config):
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["target_modules"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def train(config):
    ensure_dependencies()
    if not os.path.exists(config["data_path"]):
        raise FileNotFoundError(
            f"Training data not found: {config['data_path']}. "
            "Generate it first or pass --data-path / --config with the correct JSONL path."
        )
    os.makedirs(config["output_dir"], exist_ok=True)

    model, tokenizer = create_qlora_model(config)
    train_ds = InstructionDataset(config["data_path"], tokenizer, config["max_length"])

    use_bf16 = bool(config["bf16"] and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        num_train_epochs=config["num_train_epochs"],
        logging_steps=config["logging_steps"],
        save_strategy="steps",
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        optim=config["optim"],
        bf16=use_bf16,
        fp16=bool(torch.cuda.is_available() and not use_bf16),
        max_grad_norm=config["max_grad_norm"],
        warmup_ratio=config["warmup_ratio"],
        lr_scheduler_type=config["lr_scheduler_type"],
        report_to="none",
        load_best_model_at_end=False,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=config["gradient_checkpointing"],
        dataloader_num_workers=config["dataloader_num_workers"],
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator,
    )

    logger.info("Starting training with config: %s", json.dumps(config, ensure_ascii=False, indent=2))
    trainer.train()

    final_dir = os.path.join(config["output_dir"], "final")
    logger.info("Saving model to %s", final_dir)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    train(parse_args())
