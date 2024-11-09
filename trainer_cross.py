import glob
import gc
import numpy as np
from datasets import Dataset, concatenate_datasets
from sklearn.model_selection import KFold
from trainer import *

def build_arguments(output_path):
    train_args = build_train_arguments(output_path)
    train_args.eval_strategy='epoch'
    train_args.save_strategy='epoch'
    train_args.num_train_epochs = 2
    train_args.per_device_train_batch_size = 8
    
    lora_config = build_loraconfig()
    lora_config.lora_dropout = 0.2   # 增加泛化能力
    lora_config.r = 16
    lora_config.lora_alpha = 32
    return train_args, lora_config


# 确定最后的checkpoint目录
def find_last_checkpoint(output_dir):
    checkpoint_dirs = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
    last_checkpoint_dir = max(checkpoint_dirs, key=os.path.getctime)
    return last_checkpoint_dir

def load_model_v2(model_path, checkpoint_path='', device='cuda'):
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    # 加载lora权重
    if checkpoint_path: 
        model = PeftModel.from_pretrained(model, model_id=checkpoint_path).to(device)
    # 将基础模型的参数设置为不可训练
    for param in model.base_model.parameters():
        param.requires_grad = False
    
    # 将 LoRA 插入模块的参数设置为可训练
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
    return model

def build_trainer_v2(model, tokenizer, train_args, train_dataset, eval_dataset):
    # 开启梯度检查点时，要执行该方法
    if train_args.gradient_checkpointing:
        model.enable_input_require_grads()
    return Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],  # 早停回调
    )

def train_kfold(model_path, output_base_path, datasets, build_args_func, fold_num=5, device='cuda', last_checkpoint_path=''):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    kf = KFold(n_splits=fold_num, shuffle=True)
    results = []
    print(f"total_folds: {fold_num}")
    
    for fold, (train_index, val_index) in enumerate(kf.split(np.arange(len(datasets)))):
        print(f"fold={fold} start, train_index={train_index}, val_index={val_index}")
        train_dataset = datasets.select(train_index)
        eval_dataset = datasets.select(val_index)
        print(f"train data: {len(train_dataset)}, eval: {len(eval_dataset)}")
    
        output_path = f'{output_base_path}_{fold}'
        train_args, lora_config = build_args_func(output_path)
        if last_checkpoint_path:
            model = load_model_v2(model_path, last_checkpoint_path, device)
            print(f"fold={fold}, load model from checkpoint: {last_checkpoint_path}")
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
            model = get_peft_model(model, lora_config)
    
        model.print_trainable_parameters()
        trainer = build_trainer_v2(model, tokenizer, train_args, train_dataset, eval_dataset)
        train_result = trainer.train()
        print(f"fold={fold}, result = {train_result}")
        results.append(train_result)
        
        last_checkpoint_path = find_last_checkpoint(output_path)
        
    return results
