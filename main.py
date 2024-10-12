from data_processor import get_data
from data_preprocessor import preprocess_data
from model import get_model
from trainer import train_model
from evaluator import evaluate_model

def main():
    # 数据处理
    print("正在处理数据...")
    data_dir = './data'
    train_df, test_df = get_data(data_dir)

    # 数据预处理
    print("正在预处理数据...")
    train_inputs, train_masks, train_labels = preprocess_data(train_df)
    test_inputs, test_masks, test_labels = preprocess_data(test_df)

    # 获取模型
    print("正在初始化模型...")
    model, device = get_model()

    # 训练模型
    print("开始训练模型...")
    trained_model = train_model(model, train_inputs, train_masks, train_labels, device)

    # 评估模型
    print("正在评估模型...")
    evaluate_model(trained_model, test_inputs, test_masks, test_labels, device)

if __name__ == "__main__":
    main()
