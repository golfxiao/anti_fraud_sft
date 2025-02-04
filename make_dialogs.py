import os
import re
import json
import textwrap
import Agently
import queue
import pandas as pd
from tqdm import *
from typing import List
from concurrent.futures import ThreadPoolExecutor, ALL_COMPLETED
from queue import Queue

def load_jsonl(path):
    with open(path, 'r') as file:
        data = [json.loads(line) for line in file]
        return data

def get_files(directory, extension='.jsonl', excludes=[]):
    files = [f for f in os.listdir(directory) if f.endswith(extension) and f not in excludes]
    return files

def remove_markdown_boundary(text):
    return re.sub(r'^```json\n(.*)\n```$', r'\1', text.strip(), flags=re.DOTALL)

def print_json(obj):
    print(json.dumps(obj, indent=4, ensure_ascii=False))


def filename(path):
    filename_with_ext = os.path.basename(path)
    filename, _ = os.path.splitext(filename_with_ext)
    return filename

def change_file_extension(file_path: str, new_extension: str) -> str:
    if not new_extension:
        return file_path
    new_extension = (
        new_extension if new_extension.startswith(".") else "." + new_extension
    )
    # 将文件名和扩展名分开
    file_name, _ = os.path.splitext(file_path)
    # 构建新的文件名
    return f"{file_name}{new_extension}"

def pretty_print(text):
    wrapped_text = textwrap.fill(text, width=80)  # 设定每行的最大字符数
    print(wrapped_text)


def total_chars(strs: list):
    return sum(len(s) for s in strs)


############################## 1. 长段落切短段落 ################################

def filter_interjections(content: str) -> str:
    interjections = ['啊','呃','呀','吧','唉','嘛','噢','嗯','喂','喽','喔','哦','呵','呸','嘿','嚯','嘞','哎','噫','哟','啧','咦']
    # 构建简化的正则表达式模式，直接将所有语气词连接在一起
    pattern = "|".join(interjections)
    # 使用re.sub()函数替换语气词为空字符串
    filtered_text = re.sub(pattern, "", content)
    return filtered_text

def load_file(file_path, filter_func=None):
    file_content = ""
    with open(file_path, "r", encoding='utf8') as f:
        file_content = f.read()
    contents = file_content.split('\n\n')
    
    if filter_func:
        contents = [filter_func(content) for content in contents]
    return contents

def parse_paragraph(text: str):
    paragraph_pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ([^:]+):(.*)"
    match = re.match(paragraph_pattern, text.strip())
    if not match:
        print(f"not match paragraph pattern: {text}")
        return "", "", ""
    return match.group(1), match.group(2), match.group(3).strip()

def split_text_with_punct(text):
    if 'regex' not in split_text_with_punct.__dict__:
         # 提取标点符号为一个字符串变量
        punctuation_marks = '。？！.?!'
        # 使用变量构建正则表达式模式，如同普通字段串前面加`f`一样，允许在字符串中嵌入{}包裹的表达式。
        pattern = rf'([^{punctuation_marks}]*[{punctuation_marks}])'
        # 预编译正则表达式，大量使用时可以提高性能
        split_text_with_punct.regex = re.compile(pattern, re.UNICODE)
        
    regex = split_text_with_punct.regex
    # 使用finditer查找所有匹配项
    matches = list(regex.finditer(text))
    # 提取匹配到的段落
    paragraphs = [match.group(0).strip() for match in matches if match.group(0).strip()]
    # 找到最后一个匹配的结束位置
    last_match_end = matches[-1].end() if paragraphs else 0
    # 如果最后一个匹配的结束位置不是文本的结尾，即段落未以结束符结束，则将剩余文本作为单独一段
    if last_match_end < len(text):
        remaining_text = text[last_match_end:].strip()
        if remaining_text:
            paragraphs.append(remaining_text)
    
    return paragraphs

def split_long_paragraph(text, min_length=0, max_length=100):
    # 先分割文本
    sentences = split_text_with_punct(text)
    
    # 合并分割段落以满足长度要求
    result = []
    current_paragraph = ''
    for s in sentences:
        if len(current_paragraph) + len(s) <= max_length:  
            current_paragraph += s  
        else:
            # 如果当前段落过长，则将其添加到结果中，并开始新段落
            if len(current_paragraph) >= min_length:
                result.append(current_paragraph)
            current_paragraph = s
    
    # 添加最后一个段落
    if len(current_paragraph) >= min_length:
        result.append(current_paragraph)
    
    return result

def resplit_paragraphs(paragraphs):
    result = []
    for text in paragraphs:
        _, speaker, content = parse_paragraph(text)
        if content == "": continue
        small_paragraphs = split_long_paragraph(content, min_length=5, max_length=100)
        dialogs = [{'speaker': speaker, 'content':child_text, 'is_fraud': False} for child_text in small_paragraphs]
        result.extend(dialogs)
    return result

def build_to_rows_func(case_prefix=""):
    def to_rows(dialog: list):
        rows = []
        for item in dialog:
            data = {'case': case_prefix}
            for k in ['speaker', 'content', 'is_fraud']:
                data[k] = item[k]
            rows.append(data)
            
        return rows

    return to_rows

def build_write_func(file_name):
    def write(rows, header=False):
        df_new = pd.DataFrame(rows, index=range(len(rows)))
        df_new.to_csv(file_name, mode='a', header=header, index=False)
    return write

def generate_csv_dialogs(file_path, output_dir):
    # 加载文档，并重新切分段落
    texts = load_file(file_path, filter_interjections)
    paragraphs = resplit_paragraphs(texts)
    # 写header
    output_file_name = change_file_extension(os.path.basename(file_path), "csv")
    output_path = os.path.join(output_dir, output_file_name)
    header_df = pd.DataFrame(columns = ['case', 'speaker', 'content', 'is_fraud'])
    header_df.to_csv(output_path, mode='w', header=True, index=False)
    # 写数据
    write_fn = build_write_func(output_path)
    to_rows_fn = build_to_rows_func(filename(file_path))
    write_fn(to_rows_fn(paragraphs))
    
    print(f"generate_csv_dialogs: {output_file_name}, paragraphs count from {len(texts)} to {len(paragraphs)}")
    return output_path

################################## 2. 切分短对话集 #################################
group_column = 'case'
speaker_column = 'speaker'
content_column = 'content'
is_fraud_column = 'is_fraud'

def to_train_data(dialog:list, label, instruction, fraud_speaker=''):
    content = "\n".join(dialog)
    return {"input": content, "label": label, "fraud_speaker": fraud_speaker, "instruction": instruction}

def split_dialog(dialog: pd.DataFrame, max_length):
    small_dialogs = []
    current_segment = []
    current_label = False
    # 遍历dataframe中的所有数据行
    for _, item in dialog.iterrows():
        statement = f"{item[speaker_column]}: {item[content_column]}"
        if max_length <= 0:
            current_segment.append(statement)
        elif total_chars(current_segment) + len(statement) < max_length:
            current_segment.append(statement)
        else:
            small_dialogs.append(to_train_data(current_segment, False, "", ""))
            current_segment[:] = [statement]
        
    # 处理最后一段剩余的对话，为避免内容太碎，限制对话集在两条及以上时才有效
    if len(current_segment) >= 2:
        small_dialogs.append(to_train_data(current_segment, False, "", ""))
    return small_dialogs

def to_jsonl_of_special_length(input_path, output_path, max_length=200):
    df = pd.read_csv(input_path)
    # 按照某列的数据进行分组
    grouped = df.groupby(group_column)
    # 分割对话集，长对话按照指定长度分割成短对话
    all_dialogs = []
    for _, group in grouped:
        small_dialogs = split_dialog(group, max_length)
        all_dialogs.extend(small_dialogs)
    train_dataset = pd.DataFrame(all_dialogs)
    # orient="records" 表示dataset中的每一行是一个json对象
    # lines=True 表示每个json对象写入文件时占一行
    train_dataset.to_json(output_path, orient="records", lines=True, force_ascii=False)
    return len(all_dialogs)

def to_jsonl(input_path, output_dir):
    different_lengths = [100, 300, 500]
    out_files = []
    for max_length in different_lengths:
        output_file = f"{filename(input_path)}_train_{max_length}.jsonl"
        output_path = os.path.join(output_dir, output_file)
        dialog_num = to_jsonl_of_special_length(input_path, output_path, max_length)
        print(f"to_jsonl: convert {os.path.basename(input_path)} to {output_file}, small dialog count: {dialog_num}")
        out_files.append(output_path)
    return out_files


################################## 3. 给对话集打标签 ##############################

def build_agent_factory():
    agent_factory_4o = Agently.AgentFactory(is_debug=False)
    
    # Other settings are the same as chat mode above
    agent_factory_4o\
        .set_settings("current_model", "AzureOpenAI")\
        .set_settings("model.AzureOpenAI.auth", {
            "api_key": os.environ["OPENAI_API_KEY"],
            "api_version": "2024-05-01-preview",
            "azure_endpoint": "https://openaiquanshi1.openai.azure.com/openai/deployments/meeting-gpt4o/chat/completions?api-version=2024-05-01-preview",
        })\
        .set_settings("model.OpenAI.options", { "model": "gpt-4o" })
    return agent_factory_4o


def build_prompt_template():
    prompt_template = '''
role: 你是一个分析诈骗案例的专家，你的任务是分析对话内容中是否存在经济诈骗。
input: ${input}
instruct:
    task: 
      - 1. 分析{input}中的对话内容是否存在经济诈骗，并给出你的分析理由。
      - 2. 如果存在经济诈骗，请找出正在进行诈骗行为的发言者姓名。
    output language: ${language}
output:
    result:
        is_fraud:
            $type: boolean
            $desc: 关于{input}中是否存在明确的经济诈骗(true/false)。
        fraud_speaker:
            $type: str
            $desc: 仅当{is_fraud}=true 时找出的诈骗者姓名，如果未提及诈骗者姓名，请输出""。
        reason:
            $type: str
            $desc: 仅当{is_fraud}=true 时给出的理由。
'''
    return prompt_template
    
def predict(agent, text):
    response = agent.load_yaml_prompt(
        yaml = build_prompt_template(),
        variables={
            "input": text,
            "language": 'chinese',
        }
    ).start()
    # 如果遇到markdown格式的json，则尝试用上面封装的remove_markdown_boundary来二次处理.
    if isinstance(response, str):
        try:
            response = json.loads(remove_markdown_boundary(response))
        except json.JSONDecodeError:
            return {}
    # 确保response是一个字典
    return response.get('result', {}) if isinstance(response, dict) else {}

def run_predict(agent, dialogset, output_path, progress_queue=None):
    with open(output_path, 'a') as f:
        for i, item in enumerate(dialogset):
            dialog_input = item['input']
            prediction = predict(agent, dialog_input)
            item['label'] = prediction.get('is_fraud', None)
            item['fraud_speaker'] = prediction.get('fraud_speaker', '')
            item['reason'] = prediction.get('reason')
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            progress_queue.put(1) if progress_queue else None
    print(f"{os.path.basename(output_path)} labeled over.")

def label_dialogs(input_files, output_dir):
    agent_factory = build_agent_factory()
    data_chunks = [load_jsonl(f) for f in input_files]
    output_files = [os.path.join(output_dir, os.path.basename(f)) for f in input_files]
    total_count = sum(len(chunk) for chunk in data_chunks)
    print(f"chunks: {len(data_chunks)}, total_count: {total_count}")
    
    # 创建一个队列来传递进度信息
    progress_queue = Queue()
    pbar = tqdm(total=total_count, desc="total progress")
    with ThreadPoolExecutor(max_workers=len(data_chunks)) as executor:
        futures = [executor.submit(run_predict, agent_factory.create_agent(), chunk, output_files[i], progress_queue) for i, chunk in enumerate(data_chunks)]
        # 在主线程中轮询队列来更新进度条
        while any(not f.done() for f in futures):
            try:
                # 尝试从队列中获取进度信息
                progress = progress_queue.get(timeout=10) # 使用get_nowait避免阻塞
                pbar.update(progress)
            except queue.Empty:
                print("No item available within the timeout period.")
    # 进度条使用完后，需要关闭。
    pbar.close()
    return output_files

################################ 主流程 ##################################

def main():
    # 输入目录及文件定义
    asr_dir = "/data2/anti_fraud/dataset/docs/"
    file_names = [
        "广发煤炭_淮河能源近况交流电话会议.txt",
        "39046229_20221103_095950.txt", 
        "比亚迪专家会.stt",
    ]
    # 中间文件的输出目录定义
    csv_output_dir = "/data2/anti_fraud/dataset/fraud/csv_dialogs/"
    jsonl_output_dir = "/data2/anti_fraud/dataset/fraud/jsonl"
    labeled_output_dir = "/data2/anti_fraud/dataset/fraud/labeled"
    os.makedirs(csv_output_dir, exist_ok=True)
    os.makedirs(jsonl_output_dir, exist_ok=True)
    os.makedirs(labeled_output_dir, exist_ok=True)

    # 通过索引下标，控制一次为哪些文件生成对话集
    input_files = file_names[-1:]
    output_files = []
    
    # 主循环
    for fname in input_files:
        asr_path = os.path.join(asr_dir, fname)
        
        if not os.path.exists(asr_path):
            print(f"file_path: {asr_path} not exists")
            continue
        print(f"make dialogs for file_path: {asr_path} ...")
        
        csv_path = generate_csv_dialogs(asr_path, csv_output_dir)
        # 由 csv短段落集 转成 jsonl短对话集 
        jsonl_files = to_jsonl(csv_path, jsonl_output_dir)
        # 对 jsonl短对话集 打标签 
        labeled_files = label_dialogs(jsonl_files, labeled_output_dir)
        output_files.extend(labeled_files)
    
    print(f"make dialogs over, input: {len(input_files)} files, output {len(output_files)} files, paths as follow:")
    for file in output_files: print(file)


# 功能：输入asr，生成打好标签的短对话集
# cd到脚本目录下运行: python make_dialogs.py
if __name__ == "__main__": 
    main()
