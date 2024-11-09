import os
import json
import torch
import pandas as pd
from typing import Dict
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import Dataset
import transformers
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig, EarlyStoppingCallback
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from accelerate import Accelerator

def load_jsonl(path):
    with open(path, 'r') as file:
        data = [json.loads(line) for line in file]
        return pd.DataFrame(data)

def view_data_distribution(data_path, show_first=False):
    df = load_jsonl(data_path)
    print(f"total_count:{df.shape[0]}, true_count: {df['label'].sum()}, false_count: {(df['label']==False).sum()}")
    print(json.dumps(df.iloc[0].to_dict(), indent=4, ensure_ascii=False)) if show_first else None

def preprocess(item, tokenizer, with_reason=False, max_length=2048, instruction=None):
    system_message = "You are a helpful assistant."
    instruction = item['instruction'] if not instruction else instruction
    user_message = instruction + '\n' + item['input']
    if with_reason: 
        output = {"is_fraud":item["label"], "fraud_speaker":item["fraud_speaker"], "reason":item["reason"]}
    else:
        output = {"is_fraud":item["label"]}
        
    assistant_message = json.dumps(output, ensure_ascii=False)
    input_ids, attention_mask, labels = [], [], []
    input = tokenizer(f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)  
    response = tokenizer(assistant_message, add_special_tokens=False)
    input_ids = input["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = input["attention_mask"] + response["attention_mask"] + [1]  
    # -100是一个特殊的标记，用于指示指令部分的token不应参与损失计算
    labels = [-100] * len(input["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
    
    # 对输入长度做一个限制保护，超出截断
    return {
        "input_ids": input_ids[:max_length],
        "attention_mask": attention_mask[:max_length],
        "labels": labels[:max_length]
    }


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self, raw_data, tokenizer: transformers.PreTrainedTokenizer, with_reason=False, max_len: int = 1024, instruction=None
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.raw_data = raw_data
        self.with_reason = with_reason
        self.instruction=instruction
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if isinstance(i, list):
            ret = [preprocess(self.raw_data[j], self.tokenizer, self.with_reason, self.max_len, self.instruction) for j in i]
        elif isinstance(i, int):
            ret = [preprocess(self.raw_data[i], self.tokenizer, self.with_reason, self.max_len, self.instruction)]
        else:  
            raise TypeError("Index must be an int or a list of ints")  
            
        df = pd.DataFrame(ret)
        
        return dict(
            input_ids=df['input_ids'],
            labels=df['labels'],
            attention_mask=df['attention_mask'],
        )

def load_one_dataset(data_path, tokenizer, with_reason:bool, lazy:bool, instruction=None):
    df = load_jsonl(data_path)
    ds = Dataset.from_pandas(df)
    if lazy:
        return LazySupervisedDataset(ds, tokenizer, with_reason)
    else:
        return ds.map(
            lambda x: preprocess(x, tokenizer, with_reason=with_reason, instruction=instruction),
            remove_columns=ds.column_names)

def load_dataset(train_path, eval_path, tokenizer, with_reason=False, lazy=True, instruction=None):
    train_dataset = load_one_dataset(train_path, tokenizer, with_reason, lazy, instruction)
    eval_dataset = load_one_dataset(eval_path, tokenizer, with_reason, lazy, instruction)
    return train_dataset, eval_dataset
    
def load_model(model_path, device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    # torch_dtype控制参数加载的精度
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    return model.to(device), tokenizer

def build_loraconfig():
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False, # 训练模式
        r=8, 
        lora_alpha=16,   
        lora_dropout=0.05
    )

def build_train_arguments(output_path):
    return TrainingArguments(
        output_dir=output_path,
        per_device_train_batch_size=16,  # 每个设备（如每个GPU）的训练批次大小
        gradient_accumulation_steps=1,  # 梯度累积的步骤数，相当于增大批次大小
        log_level="warning",
        log_level_replica="warning",
        logging_steps=10,  
        logging_first_step=True, # 是否在训练的第一步就记录日志
        logging_dir=os.path.join(output_path, "logs"),
        weight_decay=0.01,     # 引入权重衰减，它会迫使模型参数保持较小的值，从而避免模型过拟合
        warmup_ratio=0.05,     # 前5%的steps用于预热学习率
        num_train_epochs=3,    # 训练的总轮数
        per_device_eval_batch_size=8,  # 每个设备（如每个GPU）的预测批次大小
        eval_strategy="steps",  # 设置评估策略为steps
        eval_on_start=False,    # 在训练开始时就进行模型评估
        eval_steps=100,  # 设置评估的步数，与保存步数一致
        save_steps=100, # 为了快速演示，这里设置10，建议你设置成100
        learning_rate=1e-4,
        bf16=True,                   # 使用混合精度来进行矩阵计算，bf16范围大精度低，在数值稳定性上比fp16表现更好，
        lr_scheduler_type="cosine",  # 使用余弦退火调度器，在训练过程中逐渐减小学习率，有助于模型稳定收敛。
        save_on_each_node=True,
        load_best_model_at_end=True, # 在训练结束时加载最佳模型
        remove_unused_columns=False, # 
        dataloader_drop_last=True,   # 抛弃最后一批迭代数据（可能数量不满足一批）
        gradient_checkpointing=True  #  启用梯度检查点以节省内存
    )

def build_trainer(model, tokenizer, train_args, lora_config, train_dataset, eval_dataset):
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    # 开启梯度检查点时，要执行该方法
    if train_args.gradient_checkpointing:
        peft_model.enable_input_require_grads()
        
    return Trainer(
        model=peft_model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],  # 早停回调
    )
