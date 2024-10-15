import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax

# 确保你的模型和tokenizer是匹配的
model_name = 'your_model_name_or_path'  # 替换为你的模型路径
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# 确保模型处于评估模式
model.eval()

def predict_spam(text):
    # 对输入文本进行编码
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
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

# 输入文本
input_text = "This is a test message."  # 替换为你想要测试的文本

# 进行预测
probability, pred_class = predict_spam(input_text)

# 输出结果
print(f"Input Text: {input_text}")
print(f"Probability of Spam: {probability:.4f}")
if pred_class == 1:
    print("Predicted as Spam")
else:
    print("Predicted as Not Spam")