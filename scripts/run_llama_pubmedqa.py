# scripts/run_llama_pubmedqa.py

import torch
import torch.nn as nn
import numpy as np
import requests
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Dataset: PubMedQA
dataset_url = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/refs/heads/master/data/ori_pqal.json"
response = requests.get(dataset_url)
data = response.json()
questions = [item["QUESTION"] for item in data.values()]
num_samples = min(10, len(questions))

# Model variants
LLAMA_MODELS = {
    "LLaMA-2-7B": "meta-llama/Llama-2-7b-hf",
    "LLaMA-2-13B": "meta-llama/Llama-2-13b-hf"
}

# MC Dropout wrapper
class BayesianLLaMA(nn.Module):
    def __init__(self, base_model):
        super(BayesianLLaMA, self).__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = self.dropout(outputs.logits)
        return logits

# No uncertainty
def generate_response(model, tokenizer, input_text):
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        pad_to_multiple_of=8
    ).to("cuda")

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=50,
            pad_token_id=tokenizer.pad_token_id
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Uncertainty with MC dropout
def estimate_uncertainty(model, tokenizer, input_text, n_samples=10):
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        pad_to_multiple_of=8
    ).to("cuda")

    predictions = []

    for _ in range(n_samples):
        with torch.no_grad():
            logits = model(inputs["input_ids"], inputs["attention_mask"])
            predictions.append(logits.cpu().numpy())

    predictions = np.array(predictions)
    mean_prediction = predictions.mean(axis=0)
    uncertainty = np.abs(predictions.std(axis=0) / (mean_prediction + 1e-6))
    return mean_prediction, uncertainty

# Metrics
def compute_cus(uncertainty): return float(np.mean(uncertainty))

def compute_zti(cus_scores, epsilon=1e-6):
    if len(cus_scores) < 2: return 1.0
    mean_cus = np.mean(cus_scores)
    std_cus = np.std(cus_scores)
    max_cus = max(np.max(cus_scores), epsilon)
    return float(max(0, min(1, 1 - (np.log1p(abs(mean_cus - std_cus)) / np.log1p(max_cus)))))

# Main execution
def run_llama_pubmedqa():
    all_results = {}

    for model_name, model_path in tqdm(LLAMA_MODELS.items(), desc="🚀 Running LLaMA Models"):
        print(f"\n====== Running {model_name} on PubMedQA ======")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
        bayesian_model = BayesianLLaMA(model).train().to("cuda")

        cus_no_uncertainty = []
        cus_with_uncertainty = []

        for i in tqdm(range(num_samples), desc=f"🔍 {model_name} Questions"):
            # 🔹 No uncertainty
            response = generate_response(model, tokenizer, questions[i])
            print(f"\nSample {i+1} - {model_name} Response: {response}")
            score_no_unc = len(response.split())
            cus_no_uncertainty.append(score_no_unc)
            print(f"Sample {i+1} - CUS (No Uncertainty): {score_no_unc}")

            # 🔹 With uncertainty
            _, uncertainty = estimate_uncertainty(bayesian_model, tokenizer, questions[i])
            score_unc = compute_cus(uncertainty)
            cus_with_uncertainty.append(score_unc)
            print(f"Sample {i+1} - CUS (With Uncertainty): {score_unc}")

        zti_no = compute_zti(cus_no_uncertainty)
        zti_with = compute_zti(cus_with_uncertainty)

        all_results[model_name] = {
            "CUS_No_Uncertainty": cus_no_uncertainty,
            "CUS_With_Uncertainty": cus_with_uncertainty,
            "ZTI_No_Uncertainty": zti_no,
            "ZTI_With_Uncertainty": zti_with
        }

    # Save to results folder
    os.makedirs("results", exist_ok=True)
    save_path = os.path.join("results", "llama_pubmedqa_results.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n Results saved to: {save_path}")
