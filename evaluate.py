import os
import json
import re
import torch
from typing import List, Dict
from tqdm import *
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc

def load_jsonl(path):
    with open(path, 'r') as file:
        data = [json.loads(line) for line in file]
        return data

def load_model(model_path, checkpoint_path='', device='cuda'):
    # 加载tokenizer, padding_side="left"用于批量填充，仅解码器架构的模型只能使用左侧填充。
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", trust_remote_code=True)
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).eval().to(device)
    # 加载lora权重
    if checkpoint_path: 
        model = PeftModel.from_pretrained(model, model_id=checkpoint_path).to(device)
    
    return model, tokenizer

def safe_loads(text, default_value=None):
    json_string = re.sub(r'^```json\n(.*)\n```$', r'\1', text.strip(), flags=re.DOTALL)
    try:  
        return json.loads(json_string)  
    except json.JSONDecodeError as e:  
        print(f"invalid json: {json_string}")
        return default_value  

def build_prompt(content, instruction=None):
    if not instruction:
        instruction = "下面是一段对话文本, 请分析对话内容是否有诈骗风险，只以json格式输出你的判断结果(is_fraud: true/false)。"
    prompt = f"{instruction}\n{content}"
    return [{"role": "user", "content": prompt}]


def predict(model, tokenizer, content, device='cuda'):
    inputs = tokenizer.apply_chat_template(build_prompt(content),
                                           add_generation_prompt=True,
                                           tokenize=True,
                                           return_tensors="pt",
                                           return_dict=True
                                           ).to(device)
    
    default_response = {'is_fraud':False}
    gen_kwargs = {"max_new_tokens": 2048, "do_sample": True, "top_k": 1}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return safe_loads(response, default_response)



def run_test(model, tokenizer, test_data, device='cuda'):
    real_labels = []
    pred_labels = []
    pbar = tqdm(total=len(test_data), desc=f'progress')
    for i, item in enumerate(test_data):
        dialog_input = item['input']
        real_label = item['label']
        
        prediction = predict(model, tokenizer, dialog_input, device)
        pred_label = prediction['is_fraud']
        
        real_labels.append(real_label)
        pred_labels.append(pred_label)
        pbar.update(1)

    pbar.close()
    return real_labels, pred_labels

def predict_batch(model, tokenizer, contents: List[str], device='cuda', instruction=None):
    prompts = [build_prompt(content, instruction) for content in contents]
    inputs = tokenizer(
        tokenizer.apply_chat_template(prompts, add_generation_prompt=True, tokenize=False),
        padding=True,
        return_tensors="pt"
    ).to(device)
    
    default_response = {'is_fraud': False}
    gen_kwargs = {"max_new_tokens": 2048, "do_sample": True, "top_k": 1}
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        responses = []
        for i in range(outputs.size(0)):
            output = outputs[i, inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(output, skip_special_tokens=True)
            responses.append(safe_loads(response, default_response))
        return responses

def run_test_batch(model, tokenizer, test_data: List[Dict], batch_size: int = 8, device='cuda'):
    real_labels = []
    pred_labels = []
    pbar = tqdm(total=len(test_data), desc=f'progress')
    
    for i in range(0, len(test_data), batch_size):
        batch_data = test_data[i:i + batch_size]
        dialog_inputs = [item['input'] for item in batch_data]
        real_batch_labels = [item['label'] for item in batch_data]
        
        predictions = predict_batch(model, tokenizer, dialog_inputs, device)
        pred_batch_labels = [prediction['is_fraud'] for prediction in predictions]
        
        real_labels.extend(real_batch_labels)
        pred_labels.extend(pred_batch_labels)
        
        pbar.update(len(batch_data))

    pbar.close()
    return real_labels, pred_labels

def precision_recall(true_labels, pred_labels, labels=None, debug=False):
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    tn, fp, fn, tp = cm.ravel()
    print(f"tn：{tn}, fp:{fp}, fn:{fn}, tp:{tp}") if debug else None
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn)/(tn + fn + tp + fp)
    return precision, recall, accuracy

def evaluate_with_model(model, tokenizer, testdata_path, device='cuda', batch=False, debug=False):
    dataset = load_jsonl(testdata_path)
    run_test_func = run_test_batch if batch else run_test
    true_labels, pred_labels = run_test_func(model, tokenizer, dataset, device=device)
    precision, recall, accuracy = precision_recall(true_labels, pred_labels, debug=debug)
    print(f"precision: {precision}, recall: {recall}, accuracy: {accuracy}")

def evaluate(model_path, checkpoint_path, testdata_path, device='cuda', batch=False, debug=False):    
    model, tokenizer = load_model(model_path, checkpoint_path, device)
    evaluate_with_model(model, tokenizer, testdata_path, device, batch, debug)