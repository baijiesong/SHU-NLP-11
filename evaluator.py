from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate_model(model, test_inputs, test_masks, test_labels, device):
    model.eval()
    test_dataset = TensorDataset(test_inputs, test_masks, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
            labels = batch[2]
            
            outputs = model(**inputs)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")

if __name__ == "__main__":
    from data_processor import get_data
    from data_preprocessor import preprocess_data
    from model import get_model
    from trainer import train_model
    
    data_dir = './data'
    train_df, test_df = get_data(data_dir)
    train_inputs, train_masks, train_labels = preprocess_data(train_df)
    test_inputs, test_masks, test_labels = preprocess_data(test_df)
    
    model, device = get_model()
    trained_model = train_model(model, train_inputs, train_masks, train_labels, device)
    
    evaluate_model(trained_model, test_inputs, test_masks, test_labels, device)
