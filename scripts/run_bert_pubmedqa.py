# scripts/run_bert_pubmedqa.py

import os
import json
import torch
import torch.nn as nn
import numpy as np
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load dataset from PubMedQA
dataset_url = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/refs/heads/master/data/ori_pqal.json"
response = requests.get(dataset_url)
data = response.json()
questions = [item["QUESTION"] for item in data.values()]
num_samples = min(10, len(questions))

# Define BERT variants
BERT_MODELS = {
    "BERT-Base": "bert-base-uncased",
    "BERT-Large": "bert-large-uncased",
    "BioBERT": "dmis-lab/biobert-base-cased-v1.1",
    "ClinicalBERT": "emilyalsentzer/Bio_ClinicalBERT",
    "RoBERTa": "roberta-base"
}

class BayesianBERT(nn.Module):
    def __init__(self, base_model):
        super(BayesianBERT, self).__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        logits = self.dropout(outputs.logits)
        return logits

def predict_no_uncertainty(model, tokenizer, input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = softmax(logits.numpy(), axis=1)
    return probs

def estimate_uncertainty_bert(model, tokenizer, input_text, n_samples=20):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
    predictions = []

    for _ in range(n_samples):
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            predictions.append(softmax(logits.numpy(), axis=1))

    predictions = np.array(predictions)
    mean_prediction = predictions.mean(axis=0)
    uncertainty = np.abs(predictions.std(axis=0) / (mean_prediction + 1e-6))
    return mean_prediction, uncertainty

def compute_cus(uncertainties):
    return float(np.mean(uncertainties))

def compute_zti(cus_scores, epsilon=1e-6):
    if len(cus_scores) < 2:
        return 1.0
    mean_cus = np.mean(cus_scores)
    std_cus = np.std(cus_scores)
    max_cus = max(np.max(cus_scores), epsilon)
    zti = 1 - (np.log1p(abs(mean_cus - std_cus)) / np.log1p(max_cus))
    return float(max(0, min(zti, 1)))

def run_bert_pubmedqa():
    results = {}

    for model_name, model_path in BERT_MODELS.items():
        print(f"\n====== Running {model_name} on PubMedQA ======")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=4)

        bayesian_model = BayesianBERT(model)
        bayesian_model.train()

        cus_scores_no_uncertainty = []
        cus_scores_with_uncertainty = []

        for i in range(num_samples):
            probs_no_uncertainty = predict_no_uncertainty(model, tokenizer, questions[i])
            cus_no_uncertainty = compute_cus(probs_no_uncertainty)
            cus_scores_no_uncertainty.append(cus_no_uncertainty)
            print(f"Sample {i+1} - {model_name} CUS (No Uncertainty): {cus_no_uncertainty}")

            _, uncertainty = estimate_uncertainty_bert(bayesian_model, tokenizer, questions[i])
            cus = compute_cus(uncertainty)
            cus_scores_with_uncertainty.append(cus)
            print(f"Sample {i+1} - {model_name} CUS (With Uncertainty): {cus}")

        zti_no = compute_zti(cus_scores_no_uncertainty)
        zti_with = compute_zti(cus_scores_with_uncertainty)

        print(f"{model_name} ZTI (No Uncertainty): {zti_no}")
        print(f"{model_name} ZTI (With Uncertainty): {zti_with}")

        results[model_name] = {
            "CUS_No_Uncertainty": [float(x) for x in cus_scores_no_uncertainty],
            "CUS_With_Uncertainty": [float(x) for x in cus_scores_with_uncertainty],
            "ZTI_No_Uncertainty": zti_no,
            "ZTI_With_Uncertainty": zti_with
        }

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/bert_pubmedqa_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nResults saved to results/bert_pubmedqa_results.json")

# Optional CLI entry
if __name__ == "__main__":
    run_bert_pubmedqa()
