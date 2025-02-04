import os
import json
import time
import random
import pandas as pd
from random import sample

def load_jsonl_files(directory):
    dataset = []
    for filename in os.listdir(directory):
        if filename.endswith('.jsonl'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = [json.loads(line) for line in file]
                dataset.extend(data)
    return dataset

def normalize_record(record):
    return {
        "instruction": "你是一个分析诈骗案例的专家，你的任务是分析下面对话内容是否存在经济诈骗(is_fraud:<bool>)，如果存在经济诈骗，请找出正在进行诈骗行为的发言者姓名(fraud_speaker:<str>)，并给出你的分析理由(reason:<str>)，最后以json格式输出。",
        # "instruction": str(record.get("instruction", "")),
        "input": str(record.get("input", "")),
        "label": bool(record.get("label", False)),
        "fraud_speaker": str(record.get("fraud_speaker", "")),
        "reason": str(record.get("reason", ""))
    }


def rebalance_dataset(dataset):
    # 按标签值分割数据集
    true_data = [d for d in dataset if d['label'] == True]
    false_data = [d for d in dataset if d['label'] == False]
    true_data_len, false_data_len = len(true_data), len(false_data)
    # 欠采样多数类，这里有两种可能：
    # 1. 如果true_data > false_data，则以false_data的数量为基准来欠采样true_data，使其与false_data的数量相等.
    # 2. 反过来，则以true_data的数量为基准来欠采样false_data，使其与true_data的数量相等。.
    if len(true_data) > len(false_data):
        true_data = sample(true_data, len(false_data))
    else:
        false_data = sample(false_data, len(true_data))
        
    print(f"balanced_true_data:{true_data_len} -> {len(true_data)}, balanced_false_data: {false_data_len} -> {len(false_data)}")
    return true_data + false_data



def split_dataset(lines, train_ratio=0.8):
    random.shuffle(lines)
    split_index = int(len(lines) * train_ratio)
    train_data = lines[:split_index]
    test_data = lines[split_index:]
    return train_data, test_data

def make_train_eval_test(dataset, train_ratio=0.8):
    train_data, temp_data = split_dataset(dataset, train_ratio)
    eval_data, test_data = split_dataset(temp_data, 0.5)
    return train_data, eval_data, test_data

def save_data(dataset, file_path):
    df = pd.DataFrame(dataset)
    df.to_json(file_path, orient="records", lines=True, force_ascii=False)
    

def main():
    start_time = time.time()
    # 设置你的输入数据集路径，应该是按长度切分好的短对话集
    input_path = "/data2/anti_fraud/dataset/fraud/jsonl"
    dataset = load_jsonl_files(input_path)
    dataset = [normalize_record(item) for item in dataset]
    print(f"all dataset length: {len(dataset)}")

    # 定义训练数据集比例，定义的数据由验证和测试集平分
    train_ratio = 0.8  
    balanced_dataset = rebalance_dataset(dataset)
    train_data, eval_data, test_data = make_train_eval_test(balanced_dataset, train_ratio=0.8)
    print(f"train_data: {len(train_data)}, eval_data: {len(eval_data)}, test_data: {len(test_data)}")

    # 定义输出路径
    output_path = "/data2/anti_fraud/dataset/fraud/train_test"
    os.makedirs(output_path, exist_ok=True)
    save_data(train_data, os.path.join(output_path, 'train20250123.jsonl'))
    save_data(eval_data, os.path.join(output_path, 'eval20250123.jsonl'))
    save_data(test_data, os.path.join(output_path, 'test20250123.jsonl'))
    print(f"dataset saved in {output_path}, use time: {time.time() - start_time:.3f} s")
    

# 功能：此脚本用于按比例切分训练集、验证集和测试集，在切分前会平衡正、反向数据条数
# cd到脚本目录下运行: python make_dataset.py
if __name__ == "__main__": 
    main()
    