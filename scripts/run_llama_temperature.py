# run_llama_temperature.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.metrics import compute_confidence_entropy, compute_zti
from pathlib import Path
import json


# Bayesian Wrapper for LLaMA
class BayesianLLaMA(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        return self.dropout(outputs.logits)


def generate_response(model, tokenizer, input_text, temperature):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            max_new_tokens=50,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def estimate_uncertainty_llama(bayesian_model, tokenizer, input_text, n_samples=10):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    predictions = []

    for _ in range(n_samples):
        with torch.no_grad():
            logits = bayesian_model(inputs["input_ids"], inputs["attention_mask"])
            predictions.append(logits.cpu().numpy())

    predictions = np.array(predictions)
    mean_pred = predictions.mean(axis=0)
    std_dev = predictions.std(axis=0)
    uncertainty = np.abs(std_dev / (mean_pred + 1e-6))

    return mean_pred, uncertainty


def compute_cus(uncertainties):
    return np.mean(uncertainties)


def run_llama_temperature():
    df = pd.read_csv("data/PRESCRIPTIONS.csv", low_memory=False)
    questions = df["DRUG_NAME_GENERIC"].dropna().astype(str).tolist()
    num_samples = min(10, len(questions))

    LLAMA_MODELS = {
        "llama-2-7b": "meta-llama/Llama-2-7b-hf"
    }

    TEMPERATURES = [1.0, 0.8, 1.2]
    results = []

    for model_name, model_path in LLAMA_MODELS.items():
        print(f"\n====== Evaluating {model_name} on MIMIC-III ======")

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cpu",
            torch_dtype=torch.float32
        )
        base_model.config.use_cache = False
        base_model.config.pretraining_tp = 1
        bayesian_model = BayesianLLaMA(base_model)

        for temp in TEMPERATURES:
            print(f"\n🔹 Temperature = {temp}")
            cus_scores = []
            entropy_scores = []

            for i in range(num_samples):
                query = questions[i]
                response = generate_response(base_model, tokenizer, query, temp)
                print(f"[{i+1}] {query} → {response}")

                # Dummy probability proxy from length
                probs = np.array([len(response.split())])
                probs = probs / np.sum(probs)

                _, uncertainty = estimate_uncertainty_llama(bayesian_model, tokenizer, query)
                cus = compute_cus(probs)
                entropy = compute_confidence_entropy(probs)

                cus_scores.append(cus)
                entropy_scores.append(entropy)

            zti = compute_zti(cus_scores)

            results.append({
                "Model": model_name,
                "Temperature": temp,
                "CUS": np.mean(cus_scores),
                "ZTI": zti,
                "Confidence Entropy": np.mean(entropy_scores)
            })

    df_result = pd.DataFrame(results)
    print("\nFinal Results:\n", df_result)

    # Save results
    Path("results").mkdir(parents=True, exist_ok=True)
    with open("results/llama_temperature_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n===== LLaMA Temperature Evaluation Completed! =====")
