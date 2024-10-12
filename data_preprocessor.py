from transformers import BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_data(df, max_length=128):
    encodings = tokenizer(df['content'].tolist(), truncation=True, padding=True, max_length=max_length)
    
    input_ids = torch.tensor(encodings['input_ids'])
    attention_mask = torch.tensor(encodings['attention_mask'])
    labels = torch.tensor(df['label'].map({'ham': 0, 'spam': 1}).tolist())
    
    return input_ids, attention_mask, labels

if __name__ == "__main__":
    from data_processor import get_data
    
    data_dir = './data'
    train_df, test_df = get_data(data_dir)
    
    train_inputs, train_masks, train_labels = preprocess_data(train_df)
    test_inputs, test_masks, test_labels = preprocess_data(test_df)
    
    print(f"训练集张量形状: {train_inputs.shape}")
    print(f"测试集张量形状: {test_inputs.shape}")
