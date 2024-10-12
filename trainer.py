from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn

def train_model(model, train_inputs, train_masks, train_labels, device, batch_size=16, num_epochs=3):
    train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * num_epochs)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
            labels = batch[2]
            
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, 平均损失: {avg_loss:.4f}")

    print("训练完成!")
    return model

if __name__ == "__main__":
    from data_processor import get_data
    from data_preprocessor import preprocess_data
    from model import get_model
    
    data_dir = './data'
    train_df, _ = get_data(data_dir)
    train_inputs, train_masks, train_labels = preprocess_data(train_df)
    model, device = get_model()
    
    trained_model = train_model(model, train_inputs, train_masks, train_labels, device)
