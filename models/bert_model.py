# models/bert_model.py

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class BayesianBERT(nn.Module):
    def __init__(self, model_name="bert-large-uncased", dropout_rate=0.3):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return self.dropout(outputs.logits)
