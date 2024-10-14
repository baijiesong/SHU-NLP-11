from transformers import BertForSequenceClassification
import torch.nn as nn
import torch

class EmailClassifier(nn.Module):
    def __init__(self, num_labels=2):
        super(EmailClassifier, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

def get_model():
    model = EmailClassifier()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model, device


if __name__ == "__main__":
    model, device = get_model()
    print(model)
    print(f"使用设备: {device}")
