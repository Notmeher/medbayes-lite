# scripts/run_openai_pubmedqa.py

import os
import time
import json
import torch.nn as nn
import numpy as np
import requests
from openai import OpenAI
from dotenv import load_dotenv

#  Load API Key
load_dotenv(os.path.join("configs", ".env"))
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(" OPENAI_API_KEY not found in environment. Check config/.env file.")

client = OpenAI(api_key=api_key)

#  Load PubMedQA Dataset
url = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/refs/heads/master/data/ori_pqal.json"
response = requests.get(url)
data = response.json()
questions = [item["QUESTION"] for item in data.values()]
num_samples = min(10, len(questions))


#  Bayesian Wrapper for OpenAI Models
class BayesianOpenAI(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

    def forward(self, prompt, n_samples=10):
        predictions = []
        for _ in range(n_samples):
            resp = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=np.random.uniform(0.7, 1.2)
            )
            predictions.append(resp.choices[0].message.content)
            time.sleep(1)
        lengths = [len(p.split()) for p in predictions]
        mean_pred = np.mean(lengths)
        uncertainty = np.abs(np.std(lengths) / (mean_pred + 1e-6))
        return mean_pred, uncertainty


def generate_response_no_uncertainty(prompt, model_name):
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=1.0
    )
    return resp.choices[0].message.content


def compute_cus(uncertainties):
    return float(np.mean(uncertainties))


def compute_zti(cus_scores, epsilon=1e-6):
    if len(cus_scores) < 2:
        return 1.0
    mean_cus = np.mean(cus_scores)
    std_cus = np.std(cus_scores)
    max_cus = max(np.max(cus_scores), epsilon)
    zti = 1 - (np.log1p(abs(mean_cus - std_cus)) / np.log1p(max_cus))
    return float(np.clip(zti, 0, 1))


#  Evaluation per model
def evaluate_model(model_name):
    print(f"\n Evaluating {model_name} on PubMedQA")
    bayesian_model = BayesianOpenAI(model_name)
    cus_no_uncertainty = []
    cus_with_uncertainty = []

    for i in range(num_samples):
        q = questions[i]

        # 🔹 No Uncertainty
        response = generate_response_no_uncertainty(q, model_name)
        print(f"Sample {i+1} - {model_name} Response: {response}")
        length = len(response.split())
        cus_no_uncertainty.append(length)

        # 🔹 With Uncertainty
        _, uncertainty = bayesian_model(q, n_samples=10)
        cus_with_uncertainty.append(uncertainty)
        print(f"Sample {i+1} - CUS (With Uncertainty): {uncertainty}")

    zti_no = compute_zti(cus_no_uncertainty)
    zti_with = compute_zti(cus_with_uncertainty)

    return {
        "CUS_No_Uncertainty": cus_no_uncertainty,
        "CUS_With_Uncertainty": cus_with_uncertainty,
        "ZTI_No_Uncertainty": zti_no,
        "ZTI_With_Uncertainty": zti_with,
    }


#  Main entry point
def run_openai_pubmedqa():
    model_list = ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-3"]  # Add "gpt-3" if supported in your API tier
    results = {}

    for model in model_list:
        try:
            results[model] = evaluate_model(model)
        except Exception as e:
            print(f" Skipping {model} due to error: {e}")

    os.makedirs("results", exist_ok=True)
    with open("results/pubmedqa_openai_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print(" All OpenAI results saved to results/pubmedqa_openai_results.json")
