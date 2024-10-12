from transformers import BertForSequenceClassification
import torch.nn as nn
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier

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

def ml_model():
    bayes_model = MultinomialNB(alpha=0.02)
    sgd_model = SGDClassifier(max_iter=25000, tol=1e-4, loss="modified_huber")

    weights = [0.5,0.5,]

    ensemble = VotingClassifier(estimators=[('mnb',bayes_model),
                                            ('sgd', sgd_model),
                                           ],
                                weights=weights, voting='soft', n_jobs=-1)

if __name__ == "__main__":
    model, device = get_model()
    print(model)
    print(f"使用设备: {device}")
