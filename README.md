
本项目是一个基于大语言模型进行欺诈文本分类微调的notebook脚本集，目标是通过sft来提高模型在欺诈文本分类方面的准确率，以达到实时检测对话内容中是否存在经济诈骗风险的目的，从而帮助用户增强警惕，降低经济损失风险。

本项目涵盖了从环境安装、数据集构造、模型评测、微调训练、量化到模型导出部署的整个流程，并且内置各个阶段的数据集。在环境搭建好的情况下，理论上只需要将相关的模型、数据集路径更改为本地路径即可运行。

## 目录

1. [认识模型](./0-认识模型.ipynb)
2. [基座模型选型](./1-基座模型选型.ipynb)
3. [生成正向数据集](./2-制作数据集：正向标签.ipynb)
4. [生成反向数据集](./3-制作数据集：反向标签.ipynb)
5. [构造训练测试集](./4-制作数据集：训练测试集.ipynb)
6. [模型评测](./5-模型评测.ipynb)
7. [lora单卡训练](./6-lora单卡训练.ipynb)
8. [lora单卡二次调优](./7-lora单卡参数调优.ipynb)
9. [批量评测改造](./8-模型评测：批量评测改造.ipynb)
10. [qlora量化微调](./10-qlora量化微调.ipynb)
11. [llamafactory多卡微调](./11-llamafactory：多卡训练.ipynb)
12. [llamafactory模型导出](./12-llamafactory：模型导出与部署.ipynb)
13. [交叉训练验证](./13-lora交叉验证.ipynb)
14. [GPTQ量化模型](./14-GPTQ量化模型.ipynb)
15. [数据校正与增强](./15-制作数据集：校正与增强.ipynb)
16. [分类原因评测改造](./16-模型评测：支持分类原因改造.ipynb)
17. [lora带原因交叉验证](./17-lora带原因交叉验证.ipynb)
18. [基于llamacpp+cpu推理](./18-基于llamacpp在cpu上推理.ipynb)

## 数据集
- [Fraud_News_Reports](./dataset/Fraud_News_Reports): 最原始的诈骗案例报道数据集。
- [csv_dialogs](./dataset/csv_dialogs): 经过chatgpt还原后的对话数据集。
- [train_test](./dataset/train_test): 经过对话重组、切割、均衡处理后的训练集、验证集和测试集。

## 附环境搭建
- [conda&pytorch环境搭建笔记](https://golfxiao.blog.csdn.net/article/details/140819506)
- [vLLM&cuda安装笔记](https://golfxiao.blog.csdn.net/article/details/140877932)
- [huggingface国内镜像](https://hf-mirror.com/)


最后，感谢您阅读这个教程。如果您觉得对您有所帮助，可以考虑请我喝杯咖啡作为鼓励😊

![a cup of tea](./img/cup_of_tea.jpg)