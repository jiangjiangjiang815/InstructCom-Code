# 这个训练数据是不带COT的版本
import json
import json
import re
import torch
import logging
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset as TorchDataset

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DirectAnswerDataset(TorchDataset):
    def __init__(self, json_path, tokenizer, max_length=1024): # 【改动1】不带CoT，输出极短，可以缩小 max_length 节省显存
        self.tokenizer = tokenizer
        self.max_length = max_length

        logger.info(f"Loading data from {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                self.data = json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                self.data = [json.loads(line) for line in f]
                
        logger.info(f"Loaded {len(self.data)} examples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        instruction = item.get('instruction', '')
        input_dict = item.get('input', {})
        
        current_community = input_dict.get('current_community', '[]')
        community_stats = input_dict.get('community_state','[]')
        candidates = input_dict.get('candidates', '[]')
            
        # 【改动2】假设不带CoT的数据集，output直接是目标节点ID，例如 "42" 或 "[42]"
        output = str(item.get('output', '')).strip()

        # 【改动3】修改 Prompt，明确要求模型“直接输出”答案，不要废话
        user_content = (
            f"{instruction}\n\n"
            f"### Current Community Nodes:\n{current_community}\n\n"
            f"### Current Community state:\n{community_stats}\n\n"
            f"### Candidate Neighbors:\n{candidates}\n\n"
            f"Please directly output the ID of the single best node to add. Do not provide any reasoning or explanation."
        )
        
        messages = [
            {"role": "system", "content": "You are a Hypergraph Community Detection Expert. You always provide direct and concise answers."},
            {"role": "user", "content": user_content}
        ]
        
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_text = prompt_text + output + self.tokenizer.eos_token

        if idx == 0:
            print(f"\n{'='*20} DEBUG: Processed Input Example {'='*20}\n{full_text}\n{'='*60}\n")
            
        tokenized_full = self.tokenizer(
            full_text, 
            truncation=True, 
            max_length=self.max_length, 
            padding=False, 
            add_special_tokens=False
        )
        input_ids = list(tokenized_full["input_ids"])
        attention_mask = list(tokenized_full["attention_mask"])
        
        tokenized_prompt = self.tokenizer(
            prompt_text, 
            truncation=True, 
            max_length=self.max_length, 
            add_special_tokens=False
        )
        prompt_len = len(tokenized_prompt["input_ids"])

        # 【逻辑不变】依然只对最终的 Output（此时只有几个Token的ID）计算 Loss
        labels = [-100] * len(input_ids)
        for i in range(prompt_len, len(input_ids)):
            labels[i] = input_ids[i]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def create_qlora_model():
    model_name = "fintune/qwen25-14b" 
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=compute_dtype, 
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto", 
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "right"
    
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=128, 
        lora_alpha=32, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def compute_metrics(eval_preds, tokenizer):
    logits, labels = eval_preds
    if isinstance(logits, tuple):
        logits = logits[0]
    
    preds = np.argmax(logits, axis=-1)
    
    decoded_preds = []
    decoded_labels = []

    for i in range(len(labels)):
        valid_label_mask = labels[i] != -100
        target_ids = labels[i][valid_label_mask]
        predict_ids = preds[i][valid_label_mask]
        
        p_text = tokenizer.decode(predict_ids, skip_special_tokens=True)
        l_text = tokenizer.decode(target_ids, skip_special_tokens=True)
        
        decoded_preds.append(p_text)
        decoded_labels.append(l_text)
    
    correct = 0
    total = len(decoded_labels)
    
    for p, l in zip(decoded_preds, decoded_labels):
        # 提取数字进行精确匹配
        p_digits = "".join(re.findall(r'\d+', p))
        l_digits = "".join(re.findall(r'\d+', l))
        if p_digits == l_digits and l_digits != "":
            correct += 1

    return {"accuracy": correct / total if total > 0 else 0}

def train():
    # 确保这个数据集里的 output 字段仅仅是节点ID，没有冗长的推理过程
    DATA_PATH = "dataset/NTU2012/525_noCOT.json" 
    OUTPUT_DIR = "output/NTU2012/525noCOT"

    # 【改动4】因为 max_length 变小了，显存压力锐减，你可以尝试增大 BATCH_SIZE 加快训练
    BATCH_SIZE = 2 # 如果显卡好，可以尝试调到 4
    GRAD_ACCUM = 8 # 相应的，梯度累积步数可以减小，保持等效 Batch Size (2*8=16) 不变
    
    model, tokenizer = create_qlora_model()
    
    # max_length 从 4096 缩减到 1024，极大节省显存
    full_dataset = DirectAnswerDataset(DATA_PATH, tokenizer, max_length=1024) 
    
    logger.info("Converting to HF Dataset...")
    hf_dataset = HFDataset.from_list([full_dataset[i] for i in range(len(full_dataset))])
    
    shuffled_dataset = hf_dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = shuffled_dataset["train"]
    test_ds = shuffled_dataset["test"]

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=5e-5,
        num_train_epochs=5,
        logging_steps=5,
        save_strategy="steps",
        save_steps=50,
        eval_steps=50,
        save_total_limit=2,
        optim="paged_adamw_32bit",
        bf16=True, 
        max_grad_norm=1.0,
        warmup_ratio=0.03,
        lr_scheduler_type="linear",
        report_to="none",
        gradient_checkpointing=True 
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, tokenizer)
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info(f"Saving model to {OUTPUT_DIR}/final")
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

if __name__ == "__main__":
    train()
