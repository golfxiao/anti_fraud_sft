import glob
import gc
import numpy as np
from datasets import Dataset, concatenate_datasets
from sklearn.model_selection import KFold
from trainer import *
from evaluate_v2 import evaluate_v2

def build_arguments(output_path):
    train_args = build_train_arguments(output_path)
    train_args.eval_strategy='epoch'
    train_args.save_strategy='epoch'
    train_args.num_train_epochs = 1
    train_args.per_device_train_batch_size = 8
    
    lora_config = build_loraconfig()
    lora_config.lora_dropout = 0.2   # 增加泛化能力
    lora_config.r = 16
    lora_config.lora_alpha = 32
    return train_args, lora_config


def find_last_checkpoint(output_dir):
    # 定位输出目录中最后一个checkpoint子目录
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
        callbacks=None,  
    )

def load_and_merge_dataset(traindata_path, evaldata_path, tokenizer):
    # 合并训练和验证集
    # train_dataset, eval_dataset = load_dataset(traindata_path, evaldata_path, tokenizer, with_reason=True, lazy=False)
    # print(f"train type: {type(train_dataset)}, eval type: {type(eval_dataset)}")
    # return concatenate_datasets([train_dataset, eval_dataset])
    return load_one_dataset(evaldata_path, tokenizer, with_reason=True, lazy=False)
    

def train_kfold(model_path, output_dir_prefix, train_path, eval_path, fold_num=5, device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    datasets = load_and_merge_dataset(train_path, eval_path, tokenizer)
    kf = KFold(n_splits=fold_num, shuffle=True)
    last_checkpoint_path = ""
    model = None
    print(f"total_folds: {fold_num}, dataset length: {len(datasets)}")
    
    for fold, (train_index, val_index) in enumerate(kf.split(np.arange(len(datasets)))):
        print(f"fold={fold} start, train_index={train_index}, val_index={val_index}")
        train_dataset = datasets.select(train_index)
        eval_dataset = datasets.select(val_index)
        print(f"train data: {len(train_dataset)}, eval: {len(eval_dataset)}")
    
        output_path = f'{output_dir_prefix}_{fold}'
        train_args, lora_config = build_arguments(output_path)
        if not model:
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
            model = get_peft_model(model, lora_config)
        else:
            # model = load_model_v2(model_path, last_checkpoint_path, device)
            # print(f"fold={fold}, load model from checkpoint: {last_checkpoint_path}")
            print("use exist model state")
            
        model.print_trainable_parameters()
        trainer = build_trainer_v2(model, tokenizer, train_args, train_dataset, eval_dataset)
        train_result = trainer.train()
        print(f"fold={fold}, result = {train_result}")
        
        last_checkpoint_path = find_last_checkpoint(output_path)
        
    return last_checkpoint_path
 
 
def export_model(base_model_path, lora_path, output_path):
    print(f"Loading the base model from {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
 
    print(f"Loading the LoRA adapter from {lora_path}")
 
    lora_model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        torch_dtype=torch.bfloat16,
    )
 
    print("Merge the LoRA")
    model = lora_model.merge_and_unload()
 
    print(f"Saving the target model to {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

def main():
    device = 'cuda'
    kfold_num = 5
    
    traindata_path = '/data2/anti_fraud/dataset/train0902.jsonl'
    evaldata_path = '/data2/anti_fraud/dataset/eval0902.jsonl'
    testdata_path = '/data2/anti_fraud/dataset/test0902.jsonl'
    
    model_path = '/data2/anti_fraud/models/modelscope/hub/Qwen/Qwen2-1___5B-Instruct'
    checkpoint_dir = '/data2/anti_fraud/models/Qwen2-1___5B-Instruct_ft_20250123'
    export_dir = "/data2/anti_fraud/models/anti_fraud_20250123_1"
    
    # 训练模型
    lora_checkpoint_path = train_kfold(model_path, checkpoint_dir, traindata_path, evaldata_path, kfold_num, device)
    
    # 导出模型, 将原始模型和Lora合并
    export_model(model_path, lora_checkpoint_path, export_dir)
    
    # 用测试数据集评估模型最终性能
    evaluate_v2(export_dir, None, testdata_path, device, debug=True)

# 命令行运行: CUDA_VISIBLE_DEVICES=2 python trainer_cross.py
if __name__ == "__main__": 
    main()
    