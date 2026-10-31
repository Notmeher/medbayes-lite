import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from tqdm import tqdm

# -------------------------
# Load MedQA dataset
# -------------------------
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

medqa_path = "/home/casl/Desktop/Elias/MedBayes/test.json"
raw_json = load_json(medqa_path)

if isinstance(raw_json, dict) and "data" in raw_json:
    medqa_data = raw_json["data"]
else:
    medqa_data = raw_json

answer_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
questions = [item["question"] for item in medqa_data]
labels = [answer_map[item["answer"]] for item in medqa_data]
choices = [item.get("options", ["A","B","C","D","E"]) for item in medqa_data]

# -------------------------
# BayesianBERT (MC Dropout)
# -------------------------
class BayesianBERT(nn.Module):
    def __init__(self, base_model, dropout_rate=0.3):
        super(BayesianBERT, self).__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        return self.dropout(outputs.logits)

def estimate_uncertainty(model, tokenizer, text, n_samples=20):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    preds = []
    model.train()
    for _ in range(n_samples):
        with torch.no_grad():
            logits = model(inputs["input_ids"], inputs["attention_mask"])
            probs = softmax(logits.squeeze().cpu().numpy(), axis=-1)
            preds.append(probs)
    preds = np.array(preds)
    mean_pred = preds.mean(axis=0)
    unc = preds.std(axis=0) / (mean_pred + 1e-6)
    return mean_pred, unc

# -------------------------
# Metrics
# -------------------------
def compute_ece(probs, labels, n_bins=15):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)
    ece = 0.0
    bins = np.linspace(0, 1, n_bins+1)
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i+1])
        if np.any(mask):
            ece += np.abs(np.mean(confidences[mask]) - np.mean(accuracies[mask])) * np.mean(mask)
    return ece

def compute_nll(logits, labels):
    return F.cross_entropy(torch.tensor(logits), torch.tensor(labels)).item()

def compute_cus(uncertainties):
    return float(np.mean(uncertainties))

def compute_zti(cus_scores, eps=1e-6):
    if len(cus_scores) < 2:
        return 1.0
    mean, std = np.mean(cus_scores), np.std(cus_scores)
    max_cus = max(np.max(cus_scores), eps)
    zti = 1 - (np.log1p(abs(mean - std)) / np.log1p(max_cus))
    return float(np.clip(zti, 0, 1))

# -------------------------
# Experiment: Varying MC Samples
# -------------------------
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=5)
bayes_model = BayesianBERT(base_model, dropout_rate=0.3)

sample_settings = [5, 10, 20, 50]
results = []

for n_samples in sample_settings:
    print(f"\nRunning MedBayes with {n_samples} MC samples...")
    cus_list, preds_list, logits_list = [], [], []

    start_time = time.time()
    for q, opts in tqdm(zip(questions, choices), total=len(questions)):
        text = q + " " + " ".join(opts)
        mean_pred, unc = estimate_uncertainty(bayes_model, tokenizer, text, n_samples=n_samples)
        cus_list.append(compute_cus(unc))
        preds_list.append(mean_pred)
        logits_list.append(mean_pred)  # mean pred as proxy logits
    runtime = time.time() - start_time

    probs = np.array(preds_list)
    logits = np.array(logits_list)

    results.append((n_samples,
        compute_ece(probs, np.array(labels)),
        compute_nll(logits, np.array(labels)),
        np.mean(cus_list),
        compute_zti(cus_list),
        f"{runtime/len(questions):.2f}s per sample"
    ))

# -------------------------
# Final Table
# -------------------------
df = pd.DataFrame(results, columns=["MC Samples (M)", "ECE ↓", "NLL ↓", "CUS ↓", "ZTI ↑", "Runtime"])
print(df.to_string(index=False))
