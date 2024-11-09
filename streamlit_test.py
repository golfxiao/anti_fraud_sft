import streamlit as st 
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
model_dir = "/data2/anti_fraud/models/modelscope/hub/Qwen/Qwen2-0___5B-Instruct"

# 定义一个函数，用于获取模型和tokenizer
@st.cache_resource
def get_model():
    # 从预训练的模型中获取tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    # 从预训练的模型中获取模型，并设置模型参数
    model = LLM(model=model_dir, gpu_memory_utilization=0.25, max_num_seqs=1) 
    return tokenizer, model

def generate(prompt):
    tokenizer, model = get_model()
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    response = model.generate([text], sampling_params)  
    return response[0].outputs[0].text
    
# 创建 Streamlit 用户界面  
st.title("AI Chat")  
prompt = st.text_area("输入：")  
if st.button("发送"):  
    if prompt:  
        generated_text = generate(prompt)
        st.text_area("AI助手：", generated_text)  
    else:  
        st.warning("请提供输入提示。")