# scripts/run_gptneo_pubmedqa.py

import torch
import torch.nn as nn
import numpy as np
import requests
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.special import softmax

def run_gptneo_pubmedqa():
    # Load dataset (PubMedQA)
    dataset_url = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/refs/heads/master/data/ori_pqal.json"
    response = requests.get(dataset_url)
    data = response.json()
    questions = [item["QUESTION"] for item in data.values()]
    num_samples = min(10, len(questions))  # limit for quick testing

    # Load GPT-Neo Model
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Bayesian GPT-Neo with MC Dropout
    class BayesianGPTNeo(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            self.dropout = nn.Dropout(p=0.3)

        def forward(self, input_ids, attention_mask):
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
            return self.dropout(outputs.logits)

    bayesian_model = BayesianGPTNeo(model)
    bayesian_model.train()

    # Generate response (no uncertainty)
    def generate_response_no_uncertainty(input_text):
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        output = model.generate(inputs["input_ids"], max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(output[0], skip_special_tokens=True)

    # Estimate uncertainty using MC dropout
    def estimate_uncertainty(input_text, n_samples=20):
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        predictions = []

        for _ in range(n_samples):
            with torch.no_grad():
                logits = bayesian_model(inputs["input_ids"], inputs["attention_mask"])
                probs = softmax(logits.cpu().numpy(), axis=-1)
                predictions.append(probs)

        predictions = np.array(predictions)
        mean_pred = predictions.mean(axis=0)
        uncertainty = np.abs(predictions.std(axis=0) / (mean_pred + 1e-6))
        return mean_pred, uncertainty

    # CUS and ZTI metrics
    def compute_cus(uncertainty):
        return np.mean(uncertainty)

    def compute_zti(cus_scores, epsilon=1e-6):
        if len(cus_scores) < 2:
            return 1.0
        mean_cus = np.mean(cus_scores)
        std_cus = np.std(cus_scores)
        max_cus = max(np.max(cus_scores), epsilon)
        zti = 1 - (np.log1p(abs(mean_cus - std_cus)) / np.log1p(max_cus))
        return max(0, min(zti, 1))

    # Run evaluation
    cus_no_uncertainty = []
    cus_with_uncertainty = []

    for i in range(num_samples):
        q = questions[i]
        print(f"\n🟦 Sample {i+1}/{num_samples}")
        
        # No uncertainty
        resp = generate_response_no_uncertainty(q)
        print("🔹 GPT-Neo Response (No Uncertainty):", resp)
        cus_len = len(resp.split())  # crude proxy
        cus_no_uncertainty.append(cus_len)
        print("CUS (No Uncertainty):", cus_len)

        # With uncertainty
        _, uncertainty = estimate_uncertainty(q)
        cus = compute_cus(uncertainty)
        cus_with_uncertainty.append(cus)
        print("CUS (With Uncertainty):", cus)

    # Compute ZTI
    zti_no = compute_zti(cus_no_uncertainty)
    zti_with = compute_zti(cus_with_uncertainty)

    # Save results
    results = {
        "CUS_No_Uncertainty": [float(x) for x in cus_no_uncertainty],
        "CUS_With_Uncertainty": [float(x) for x in cus_with_uncertainty],
        "ZTI_No_Uncertainty": float(zti_no),
        "ZTI_With_Uncertainty": float(zti_with),
    }

    os.makedirs("results", exist_ok=True)
    output_path = "results/pubmedqa_gptneo_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n Results saved to {output_path}")

# For manual testing
if __name__ == "__main__":
    run_gptneo_pubmedqa()
