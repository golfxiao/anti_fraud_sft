#!/bin/bash

# 设置要使用的GPU
export CUDA_VISIBLE_DEVICES=2

# 定义日志目录  
LOG_FILE="/data2/xiaoguanghua/log/train_$(date +%Y%m%d).log" 

# 执行训练并将输出重定向到日志文件
nohup python trainer_cross.py > "$LOG_FILE" 2>&1 &

echo "Training runs in the background and you can view the log here: $LOG_FILE"