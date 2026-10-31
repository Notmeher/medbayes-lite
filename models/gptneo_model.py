# models/gptneo_model.py

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

class BayesianGPTNeo(nn.Module):
    def __init__(self, model_name="EleutherAI/gpt-neo-125M", dropout_rate=0.3):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        return self.dropout(outputs.logits)

def generate_response_no_uncertainty(model, tokenizer, input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        output_sequences = model.generate(
            inputs["input_ids"],
            max_new_tokens=50,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output_sequences[0], skip_special_tokens=True)

def estimate_uncertainty_gptneo(bayesian_model, tokenizer, input_text, n_samples=20):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids, attention_mask = inputs["input_ids"], inputs.get("attention_mask", None)
    predictions = []

    bayesian_model.train()  # Enable dropout
    for _ in range(n_samples):
        with torch.no_grad():
            logits = bayesian_model(input_ids, attention_mask)
            predictions.append(logits.detach().cpu().numpy())

    predictions = np.array(predictions)
    mean_prediction = predictions.mean(axis=0)
    uncertainty = abs(predictions.std(axis=0) / (mean_prediction + 1e-6))
    return mean_prediction, uncertainty
