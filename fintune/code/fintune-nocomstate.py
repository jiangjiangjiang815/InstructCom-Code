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
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset as TorchDataset
from datasets import load_dataset

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InstructionDataset(TorchDataset):
    def __init__(self, json_path, tokenizer, max_length=4096):
        # 将json文件中的数据读入到self.data中
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        logger.info(f"Loading JSONL data from {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))
        logger.info(f"Loaded {len(self.data)} examples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # 提取数据字段
        instruction = item.get('instruction', '')   # 任务指令的文本
        input_dict = item.get('input', {})  # 当前上下文
        
        current_community = input_dict.get('current_community', '[]')   # 当前社区
        # 【修改点 1】：删除了原本对 community_state 的提取,增加对local_hypergraph的提取
        candidates = input_dict.get('candidates', '[]') # 候选节点
        local_hypergraph = input_dict.get('local_hypergraph', 'No local hypergraph structure provided.')
        # 输出部分用来计算损失
        output = str(item.get('output', '')).strip()

        # Output 末尾加上 eos_token 告诉模型生成结束
        output_text = output + self.tokenizer.eos_token
        output_ids = self.tokenizer(output_text, add_special_tokens=False)["input_ids"]

        # prompt 最大可用长度
        max_prompt_len = self.max_length - len(output_ids)

        # 不改变原 prompt 内容，只把 local_hypergraph 单独作为可截断部分
        prefix = (
            f"{instruction}\n\n"
            f"### Current Community Nodes:\n{current_community}\n\n"
            f"### Candidate Neighbors:\n{candidates}\n\n"
            f"### Local Hypergraph Structure (1-hop):\n"
        )

        suffix = (
            f"\n\n"
            f"Please directly output the final decision without any reasoning or explanation."
        )

        # 先计算固定部分的长度
        empty_user_content = prefix + "" + suffix
        empty_messages = [
            {"role": "system", "content": "You are a Hypergraph Community Detection Expert."},
            {"role": "user", "content": empty_user_content}
        ]
        empty_prompt_text = self.tokenizer.apply_chat_template(
            empty_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        empty_prompt_ids = self.tokenizer(empty_prompt_text, add_special_tokens=False)["input_ids"]

        # local_hypergraph 的 token 预算
        graph_budget = max_prompt_len - len(empty_prompt_ids)

        if graph_budget <= 0:
            local_hypergraph = ""
        else:
            graph_ids = self.tokenizer(local_hypergraph, add_special_tokens=False)["input_ids"]
            if len(graph_ids) > graph_budget:
                graph_ids = graph_ids[:graph_budget]
                local_hypergraph = self.tokenizer.decode(graph_ids, skip_special_tokens=True)

        # 重新组装完整 user_content，文本结构与原来一致
        user_content = prefix + local_hypergraph + suffix

        messages = [
            {"role": "system", "content": "You are a Hypergraph Community Detection Expert."},
            {"role": "user", "content": user_content}
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        # 保险：如果还超长，说明固定部分本身已经太长
        # 这种情况下不要再截尾部指令，建议直接报错或跳过样本
        if len(prompt_ids) + len(output_ids) > self.max_length:
            raise ValueError(
                f"Prompt fixed parts are too long: prompt={len(prompt_ids)}, "
                f"output={len(output_ids)}, max={self.max_length}"
            )
        # 5. 拼接最终的 IDs 和掩码
        input_ids = prompt_ids + output_ids
        attention_mask = [1] * len(input_ids)
        
        # 6. 构造 Labels：Prompt 部分完全屏蔽（-100），Output 部分计算 Loss
        labels = [-100] * len(prompt_ids) + output_ids

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
def create_qlora_model():
    # 该函数用于加载qwen14B模型并配置LoRA
    model_name = "fintune/qwen25-14b" 

    # 计算精度设置，14B 模型显存压力大，务必使用 bf16
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 开启 4-bit 量化加载
        bnb_4bit_quant_type="nf4",  # 使用 NormalFloat4 数据类型
        bnb_4bit_compute_dtype=compute_dtype,   # 计算精度设置
        bnb_4bit_use_double_quant=True
    )

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",  # 自动分配显卡
        trust_remote_code=True,
    )

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "right"
    
    model = prepare_model_for_kbit_training(model)

    # 调整 LoRA 参数适配 Qwen2 架构
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

def train():
    # 注意：确保此处使用的是你已经移除了 community_state 后的新 JSON 文件路径
    DATA_PATH = "dataset/contact-primary-school/414_nocom.json"   
    OUTPUT_DIR = "output/contact-primary/nocomstate"   

    BATCH_SIZE = 2 
    GRAD_ACCUM = 8 
    
    model, tokenizer = create_qlora_model()

    # 构造训练数据集
    train_ds = InstructionDataset(DATA_PATH, tokenizer, 2048)

    # ---------- 训练参数配置 ----------
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=5e-5, 
        num_train_epochs=5, 
        logging_steps=5,
        save_strategy="steps",
        save_steps=50,  
        save_total_limit=2,
        optim="paged_adamw_8bit", 
        bf16=True,
        max_grad_norm=1.0,
        warmup_ratio=0.03,
        lr_scheduler_type="linear", 
        report_to="none",
        load_best_model_at_end=False,   
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
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
        data_collator=data_collator
    )

    logger.info("Starting training...")
    trainer.train()

    # 保存最终模型
    logger.info(f"Saving model to {OUTPUT_DIR}/final")
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

if __name__ == "__main__":
    train()