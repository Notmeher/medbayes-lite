# scripts/run_bert_variants.py

import os
import json
import numpy as np
import torch
from scipy.special import softmax
from models.bert_model import BayesianBERT
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.metrics import compute_cus, compute_zti
from utils.data_loader import load_jsonl

# BERT model variants
BERT_MODELS = {
    "BERT-Large": "bert-large-uncased",
    "BioBERT": "dmis-lab/biobert-base-cased-v1.1",
    "ClinicalBERT": "emilyalsentzer/Bio_ClinicalBERT",
    "RoBERTa": "roberta-base"
}

# Inference without Bayesian estimation
def predict_no_uncertainty(model, tokenizer, input_text):
    model.eval()
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = softmax(logits.detach().numpy(), axis=1)
    return probs

# Monte Carlo Dropout-based estimation
def estimate_uncertainty_bert(model, tokenizer, input_text, n_samples=20):
    model.train()
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
    predictions = []

    for _ in range(n_samples):
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions.append(logits.numpy())

    predictions = np.array(predictions)
    mean_prediction = predictions.mean(axis=0)
    uncertainty = np.abs(predictions.std(axis=0) / (mean_prediction + 1e-6))
    return mean_prediction, uncertainty

# Main runner
def run_bert_models():
    data_path = "data/questions.jsonl"
    result_path = "results/bert_result.json"

    os.makedirs("results", exist_ok=True)
    data = load_jsonl(data_path)
    questions = [item["question"] for item in data]
    num_samples = min(10, len(questions))

    results = {}

    for model_name, model_path in BERT_MODELS.items():
        print(f"\n====== Running {model_name} ======")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        base_model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=4)
        bayesian_model = BayesianBERT(model_name=model_path)
        bayesian_model.model = base_model

        cus_scores_no_uncertainty = []
        cus_scores_with_uncertainty = []

        for i in range(num_samples):
            # No Uncertainty
            probs = predict_no_uncertainty(base_model, tokenizer, questions[i])
            cus_no = compute_cus(probs)
            cus_scores_no_uncertainty.append(float(cus_no))
            print(f"[{i+1}] {model_name} - CUS (No Uncertainty): {cus_no:.4f}")

            # With Uncertainty
            _, unc = estimate_uncertainty_bert(bayesian_model, tokenizer, questions[i])
            cus_with = compute_cus(unc)
            cus_scores_with_uncertainty.append(float(cus_with))
            print(f"[{i+1}] {model_name} - CUS (With Uncertainty): {cus_with:.4f}")

        zti_no = compute_zti(cus_scores_no_uncertainty)
        zti_with = compute_zti(cus_scores_with_uncertainty)

        results[model_name] = {
            "CUS_No_Uncertainty": cus_scores_no_uncertainty,
            "CUS_With_Uncertainty": cus_scores_with_uncertainty,
            "ZTI_No_Uncertainty": float(zti_no),
            "ZTI_With_Uncertainty": float(zti_with)
        }

    # Save to JSON
    with open(result_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n Results saved to: {result_path}")
