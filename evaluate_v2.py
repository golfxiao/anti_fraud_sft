import jieba
from typing import List, Dict
from rouge import Rouge
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  
from evaluate import *

jieba.setLogLevel(jieba.logging.INFO)

def make_str(data):  
    return ','.join(map(str, data)) if isinstance(data, list) else data 

def make_results(dataset):
    return {
        "labels": [item.get('label', item.get('is_fraud', False)) for item in dataset],
        "speakers": [make_str(item.get('fraud_speaker', '')) for item in dataset],
        "reasons": [item.get('reason', '') for item in dataset],
    }


def run_test_batch_v2(model, tokenizer, test_data: List[Dict], batch_size: int = 8, device='cuda'):
    real_outputs, pred_outputs = [], []
    instruction = test_data[0].get('instruction', None)
    pbar = tqdm(total=len(test_data), desc=f'progress')
    
    for i in range(0, len(test_data), batch_size):
        batch_data = test_data[i:i + batch_size]
        batch_inputs = [item['input'] for item in batch_data]
        predictions = predict_batch(model, tokenizer, batch_inputs, device, instruction=instruction)
        
        real_outputs.extend(batch_data)
        pred_outputs.extend(predictions)
        
        pbar.update(len(batch_data))

    pbar.close()
    return make_results(real_outputs), make_results(pred_outputs)

# 中文分词
def tokenize(text):
    return ' '.join(list(jieba.cut(text, cut_all=False)))

# Rouge（Recall-Oriented Understudy for Gisting Evaluation）主要用于评估生成文本的质量，衡量生成的文本与参考文本的重合程度。
def rouge_score(predictions, realitys):
    rouge = Rouge()
    return rouge.get_scores(predictions, realitys, avg=True)

# 中文序列的rouge分数计算封装
def rouge_zh_score(pred_dataset, real_dataset):
    # 去掉两个集合中的空串，如果字符串集合中有空串时，会导致计算Rouge分数时报“Hypothesis is empty.”的错误，
    filtered_pairs = [(p, r) for p, r in zip(pred_dataset, real_dataset) if p.strip() != "" and r.strip() != ""]  
    filtered_predictions, filtered_realitys = zip(*filtered_pairs) if filtered_pairs else ([], [])  
    # 进行中文分词，因为Rouge分数的计算依赖于分词后的单词序列。
    predictions = [tokenize(text) for text in filtered_predictions]
    realitys = [tokenize(text) for text in filtered_realitys]
    score = rouge_score(predictions, realitys)
    return score['rouge-l']


def evaluate_with_model_v2(model, tokenizer, testdata_path, device='cuda', debug=False):
    dataset = load_jsonl(testdata_path)
    real_result, pred_result = run_test_batch_v2(model, tokenizer, dataset, device=device)
    
    precision, recall, accuracy = precision_recall(real_result["labels"], pred_result["labels"], debug=debug)
    print(f"is_fraud字段指标: \nprecision: {precision}, recall: {recall}, accuracy: {accuracy}")
    
    accuracy = accuracy_score(real_result["speakers"], pred_result["speakers"])  
    print(f"fraud_speaker字段指标: \naccuracy: {accuracy}")

    score = rouge_zh_score(pred_result["reasons"], real_result["reasons"])
    print(f"reason字段指标: \nprecision: {score['p']}, recall: {score['r']}, f1-score: {score['f']}")
    
    return real_result, pred_result
    

def evaluate_v2(model_path, checkpoint_path, testdata_path, device='cuda', debug=False):    
    model, tokenizer = load_model(model_path, checkpoint_path, device)
    return evaluate_with_model_v2(model, tokenizer, testdata_path, device, debug)