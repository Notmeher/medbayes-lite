# scripts/run_gptneo.py

import json
import torch
from tqdm import tqdm
from models.gptneo_model import BayesianGPTNeo, generate_response_no_uncertainty, estimate_uncertainty_gptneo
from utils.data_loader import load_questions_from_jsonl
from utils.metrics import compute_cus, compute_zti

def run_gptneo():
    # Load data
    data_path = "data/questions.jsonl"
    questions = load_questions_from_jsonl(data_path)
    num_samples = min(10, len(questions))  # Limit for benchmarking

    # Initialize model
    model = BayesianGPTNeo()
    tokenizer = model.tokenizer
    base_model = model.base_model

    cus_no_uncertainty = []
    cus_with_uncertainty = []

    print("\n Running GPT-Neo Benchmark with Bayesian Inference...\n")

    for i in tqdm(range(num_samples), desc="Running GPT-Neo"):
        question = questions[i]
        input_text = question["question"]  # Fix: extract the actual string

        # Without uncertainty
        response = generate_response_no_uncertainty(base_model, tokenizer, input_text)
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            logits = base_model(**inputs).logits
        cus_no_u = compute_cus(logits.detach().numpy())
        cus_no_uncertainty.append(cus_no_u)

        # With uncertainty (MC Dropout)
        _, uncertainty = estimate_uncertainty_gptneo(model, tokenizer, input_text)
        cus_u = compute_cus(uncertainty)
        cus_with_uncertainty.append(cus_u)

        # Print with scientific format
        print(f"[{i+1}] GPT-Neo CUS - NoU: {cus_no_u:.3e} | Uncertainty: {cus_u:.3e}")

    # Compute ZTI
    zti_no_u = compute_zti(cus_no_uncertainty)
    zti_with_u = compute_zti(cus_with_uncertainty)

    # Save results
    results = {
        "Model": "GPT-Neo",
        "CUS_No_Uncertainty": [float(x) for x in cus_no_uncertainty],
        "CUS_With_Uncertainty": [float(x) for x in cus_with_uncertainty],
        "ZTI_No_Uncertainty": float(zti_no_u),
        "ZTI_With_Uncertainty": float(zti_with_u),
    }

    save_path = "results/gptneo_result.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n Results saved to {save_path}")
