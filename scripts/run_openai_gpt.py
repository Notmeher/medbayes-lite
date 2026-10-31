# scripts/run_openai_gpt.py

import os
import numpy as np
from models.openai_model import OpenAIModel
from utils.data_loader import load_jsonl
from utils.metrics import compute_zti
import json
import time

def run_openai():
    model = OpenAIModel(model_name="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))

    data = load_jsonl("data/questions.jsonl")
    questions = [item["question"] for item in data][:10]

    all_uncertainties = []
    generations = []

    for i, q in enumerate(questions):
        lengths = []
        for _ in range(10):
            response = model.generate(q, temperature=np.random.uniform(0.7, 1.2))
            generations.append(response)
            lengths.append(len(response.split()))
            time.sleep(1)  # Avoid rate limits
        mean_pred = np.mean(lengths)
        uncertainty = np.std(lengths) / (mean_pred + 1e-6)
        all_uncertainties.append(uncertainty)
        print(f"Sample {i+1} | OpenAI GPT Uncertainty: {uncertainty:.4f}")

    zti_score = compute_zti(all_uncertainties)
    print("\nOpenAI GPT Zero-shot Trustworthiness Index (ZTI):", zti_score)

    results = {
        "Generations": generations,
        "CUS_Scores": all_uncertainties,
        "ZTI": zti_score
    }

    os.makedirs("results", exist_ok=True)
    with open("results/openai_result.json", "w") as f:
        json.dump(results, f, indent=2)
