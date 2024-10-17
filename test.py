import torch
from transformers import AutoTokenizer,DebertaForSequenceClassification
from torch.nn.functional import softmax
import os
import time
import pandas as pd
def load_model(model_path, device):
    # 加载预训练的 BERT 模型结构，设置分类任务的输出标签数量为 2（可以根据你的任务调整）
    model = DebertaForSequenceClassification.from_pretrained("microsoft/deberta-base", num_labels=2)
    # 加载自定义的模型权重
    model.load_state_dict(torch.load(model_path),strict=False)
    # 将模型移动到 GPU 或 CPU
    model.to(device)
    # 设置模型为评估模式（禁用dropout等）
    model.eval()
    return model

def predict_spam(text):
    # 对输入文本进行编码
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        # 获取模型的输出
        outputs = model(**inputs)
        logits = outputs.logits
        # 使用softmax函数获取概率
        probabilities = softmax(logits, dim=1)
        # 获取预测类别
        pred_class = torch.argmax(probabilities, dim=1)
        # 返回概率和预测类别
        return probabilities[0][1].item(), pred_class.item()

def read_file(file_path):
    """尝试用多种编码读取文件，处理编码错误"""
    encodings = ['utf-8', 'latin-1', 'ISO-8859-1', 'GBK']
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as file:
                return file.read()
        except UnicodeDecodeError:
            print(f"使用编码 {enc} 读取失败，尝试其他编码...")
    raise ValueError(f"无法读取文件: {file_path}，请检查文件编码。")


if __name__ == '__main__':
    model_path = 'E://github_project//deberta_model_v1.pth'  # 你保存的模型权重文件
    device = torch.device('cpu')
    # 加载模型
    model = load_model(model_path, device)
    
    # 加载 BERT 预训练的 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
    
    csv_file_path = 'E://github_project//SHU-NLP-11//spam.csv'  # 替换为你的CSV文件路径
    df = pd.read_csv(csv_file_path)
    true_labels = df['v1'].apply(lambda x: 1 if x == 'Ham' else 0).tolist()
    texts = df['v2'].tolist()
    labels = df['v1'].tolist()
    predictions = []
    success = 0
    total = len(true_labels)  # 样本总数
    print(f"总数：{total}")
    # 对每一条文本进行预测
    for i, text in enumerate(texts):
        prob, pred_class = predict_spam(text)

        # 根据预测类别判断是 Ham 还是 Spam
        if pred_class == 0:
            predictions.append("Spam")
            print(f"预测结果: 垃圾邮件 (Spam)")
        else:
            predictions.append("Spam")
            print(f"预测结果: 正常邮件 (Ham)")
        # 比较预测结果与真实标签，增加 success 计数
        print(f"正确结果：{labels[i]}")
        if pred_class != true_labels[i]:
            success += 1
        print(f"正确预测的数量：{success}")
    # 计算准确率
    accuracy = success / total
    print(f"模型在CSV文件上的预测准确率: {accuracy:.4f}")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #输入文本
    # input_text = input("请输入您要处理的文件：")

    # try:
    #     # input_text = read_file(file_path)

    #     # 打印读取的文件内容
    #     print(f"文件内容:\n{input_text}")

    #     # 进行预测
    #     probability, pred_class = predict_spam(input_text)
    #     # 输出结果
    #     print(f"Input Text: {input_text}")
    #     print(f"Probability of Ham: {probability:.4f}")
    #     print(f"class: {pred_class}")
    #     if pred_class == 1:
    #         print("Predicted as Ham")
    #     else:
    #         print("Predicted as Spam")

    # except FileNotFoundError:
    #     print(f"文件 '{input_text}' 未找到，请检查路径是否正确。")
    # except Exception as e:
    #     print(f"读取文件时发生错误: {e}")

    # # 提示用户输入要处理的文件夹路径
    # folder_path = input("请输入要处理的文件夹路径：")
    # # 初始化统计变量
    # spam_count = 0
    # ham_count = 0
    # total_files = 0
    # sum_count = 0
    # try:
    #     # 遍历文件夹中的所有文件
    #     for filename in os.listdir(folder_path):
    #         file_path = os.path.join(folder_path, filename)
            
    #         # 检查是否为文件（忽略子文件夹）
    #         if os.path.isfile(file_path):
    #             total_files += 1
                
    #             # 读取文件内容
                
    #             input_text = read_file(file_path) 

    #             # 打印读取的文件内容
    #             #print(f"正在处理文件: {filename}")
    #             #print(f"文件内容:\n{input_text}")

    #             # 进行预测
    #             probability, pred_class = predict_spam(input_text)

    #             # 根据预测结果统计垃圾邮件和正常邮件的数量
    #             if pred_class == 1:
    #                 spam_count += 1
    #                 print(f"预测结果: 垃圾邮件 (Spam)")
    #             else:
    #                 ham_count += 1
    #                 print(f"预测结果: 正常邮件 (Ham)")
    #             sum_count+=1
    #             print(f"处理的了第{sum_count}封\n")
    #             time.sleep(0.2)
    #             # print(f"垃圾邮件概率: {probability:.4f}\n")

    #     # 输出最终统计结果
    #     print(f"\n处理完成！总文件数: {total_files}")
    #     print(f"垃圾邮件个数: {spam_count}")
    #     print(f"正常邮件个数: {ham_count}")

    # except FileNotFoundError:
    #     print(f"文件夹 '{folder_path}' 未找到，请检查路径是否正确。")
    # except Exception as e:
    #     print(f"处理文件时发生错误: {e}")