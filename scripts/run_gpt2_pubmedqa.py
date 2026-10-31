# scripts/run_gpt2_pubmedqa.py

import torch
import torch.nn as nn
import numpy as np
import requests
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

def run_gpt2_pubmedqa():
    # Load dataset (PubMedQA)
    url = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/refs/heads/master/data/ori_pqal.json"
    response = requests.get(url)
    data = response.json()

    questions = [item["QUESTION"] for item in data.values()]
    num_samples = min(10, len(questions))

    # Load GPT-2 model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Bayesian GPT Class
    class BayesianGPT(nn.Module):
        def __init__(self, base_model):
            super(BayesianGPT, self).__init__()
            self.base_model = base_model
            self.dropout = nn.Dropout(p=0.3)

        def forward(self, input_ids, attention_mask):
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
            return self.dropout(outputs.logits)

    bayesian_model = BayesianGPT(model)

    # Generate response (No Uncertainty)
    def generate_response_no_uncertainty(input_text):
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            output = model.generate(inputs["input_ids"], max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(output[0], skip_special_tokens=True)

    # Estimate Uncertainty (MC Dropout)
    def estimate_uncertainty_gpt(input_text, n_samples=20):
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        predictions = []

        for _ in range(n_samples):
            with torch.no_grad():
                logits = bayesian_model(inputs["input_ids"], inputs["attention_mask"])
                predictions.append(logits.cpu().numpy())

        predictions = np.array(predictions)
        mean_pred = predictions.mean(axis=0)
        uncertainty = np.abs(predictions.std(axis=0) / (mean_pred + 1e-6))
        return mean_pred, uncertainty

    # Clinical Uncertainty Score (CUS)
    def compute_cus(uncertainties):
        return np.mean(uncertainties)

    # Trustworthiness Index (ZTI)
    def compute_zti(cus_scores, epsilon=1e-6):
        if len(cus_scores) < 2:
            return 1.0
        mean_cus = np.mean(cus_scores)
        std_cus = np.std(cus_scores)
        max_cus = max(np.max(cus_scores), epsilon)
        zti = 1 - (np.log1p(abs(mean_cus - std_cus)) / np.log1p(max_cus))
        return max(0, min(zti, 1))

    # Run Evaluation
    cus_scores_no_uncertainty = []
    cus_scores_with_uncertainty = []

    for i in range(num_samples):
        q = questions[i]

        # No Uncertainty
        response = generate_response_no_uncertainty(q)
        print(f"Sample {i+1} - GPT-2 Response:", response)
        with torch.no_grad():
            logits = model(tokenizer(q, return_tensors="pt").input_ids.to(device)).logits
            cus_no = compute_cus(logits.detach().cpu().numpy())
        cus_scores_no_uncertainty.append(cus_no)

        # With Uncertainty
        _, uncertainty = estimate_uncertainty_gpt(q)
        cus_with = compute_cus(uncertainty)
        cus_scores_with_uncertainty.append(cus_with)
        print(f"Sample {i+1} - CUS (With Uncertainty): {cus_with}")

    # Compute ZTI
    zti_no = compute_zti(cus_scores_no_uncertainty)
    zti_with = compute_zti(cus_scores_with_uncertainty)

    # Summary
    results = {
        "CUS_No_Uncertainty": cus_scores_no_uncertainty,
        "CUS_With_Uncertainty": cus_scores_with_uncertainty,
        "ZTI_No_Uncertainty": zti_no,
        "ZTI_With_Uncertainty": zti_with,
    }

    print("\n===== Final PubMedQA GPT-2 Report =====")
    for k, v in results.items():
        print(f"{k}: {v}")

    # Save results (convert to native types)
    results_serializable = {
        k: [float(vv) for vv in v] if isinstance(v, list) else float(v)
        for k, v in results.items()
    }

    os.makedirs("results", exist_ok=True)
    with open("results/pubmedqa_gpt2_results.json", "w") as f:
        json.dump(results_serializable, f, indent=4)

    print(" Results saved to results/pubmedqa_gpt2_results.json")
