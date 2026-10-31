# scripts/run_llama_prescription.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load MIMIC-III Dataset
df = pd.read_csv("data/PRESCRIPTIONS.csv", low_memory=False)
questions = df["DRUG_NAME_GENERIC"].dropna().astype(str).tolist()
num_samples = min(10, len(questions))

# Define LLaMA-2 Models
LLAMA_MODELS = {
    "LLaMA-2-7B": "meta-llama/Llama-2-7b-hf",
    "LLaMA-2-13B": "meta-llama/Llama-2-13b-hf"
}

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Bayesian LLaMA Class
class BayesianLLaMA(nn.Module):
    def __init__(self, base_model):
        super(BayesianLLaMA, self).__init__()
        self.base_model = base_model
        self.device = next(base_model.parameters()).device

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

# No-Uncertainty Generation
def generate_response_no_uncertainty(model, tokenizer, input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model.generate(inputs["input_ids"], max_new_tokens=50, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Estimate Uncertainty
def estimate_uncertainty_llama(model, tokenizer, input_text, n_samples=10):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    predictions = []

    for _ in range(n_samples):
        with torch.no_grad():
            logits = model(inputs["input_ids"], inputs["attention_mask"])
            dropout_logits = nn.functional.dropout(logits, p=0.3, training=True)
            predictions.append(dropout_logits.cpu().numpy())

    predictions = np.array(predictions)
    mean_pred = predictions.mean(axis=0)
    uncertainty = np.abs(predictions.std(axis=0) / (mean_pred + 1e-6))
    return mean_pred, uncertainty

# Metrics
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

# Evaluation
results = {}

for model_name, model_path in LLAMA_MODELS.items():
    print(f"\n====== Running {model_name} on MIMIC-III (GPU) ======")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )
    bayesian_model = BayesianLLaMA(model)

    cus_scores_no_unc = []
    cus_scores_with_unc = []

    for i in range(num_samples):
        response = generate_response_no_uncertainty(model, tokenizer, questions[i])
        print(f"Sample {i+1} - {model_name} Response:", response)
        cus_no = len(response.split())
        cus_scores_no_unc.append(cus_no)

        _, unc = estimate_uncertainty_llama(bayesian_model, tokenizer, questions[i])
        cus_with = compute_cus([unc])
        cus_scores_with_unc.append(cus_with)
        print(f"Sample {i+1} - {model_name} CUS (With Uncertainty): {cus_with}")

    zti_no = compute_zti(cus_scores_no_unc)
    zti_with = compute_zti(cus_scores_with_unc)

    print(f"{model_name} ZTI (No Uncertainty): {zti_no}")
    print(f"{model_name} ZTI (With Uncertainty): {zti_with}")

    results[model_name] = {
        "CUS_No_Uncertainty": cus_scores_no_unc,
        "CUS_With_Uncertainty": cus_scores_with_unc,
        "ZTI_No_Uncertainty": zti_no,
        "ZTI_With_Uncertainty": zti_with,
    }

# Final Report
print("\n===== Final Comparison Report =====")
for model_name, metrics in results.items():
    print(f"\nModel: {model_name}")
    print("CUS (No Uncertainty):", metrics["CUS_No_Uncertainty"])
    print("CUS (With Uncertainty):", metrics["CUS_With_Uncertainty"])
    print("ZTI (No Uncertainty):", metrics["ZTI_No_Uncertainty"])
    print("ZTI (With Uncertainty):", metrics["ZTI_With_Uncertainty"])

# Save to results/
os.makedirs("results", exist_ok=True)
with open("results/llama_prescription_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Results saved to results/llama_prescription_results.json")
