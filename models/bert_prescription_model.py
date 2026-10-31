# models/bert_prescription_model.py

import torch
import torch.nn as nn

class BayesianBERT(nn.Module):
    def __init__(self, base_model):
        super(BayesianBERT, self).__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(p=0.3)  # Bayesian MC dropout

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        return self.dropout(outputs.logits)
