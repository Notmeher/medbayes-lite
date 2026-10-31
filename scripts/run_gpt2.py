# scripts/run_gpt2.py

import torch
import numpy as np
import json
from tqdm import tqdm

from models.gpt2_model import BayesianGPT2, generate_response_no_uncertainty, estimate_uncertainty_gpt2

from utils.data_loader import load_questions_from_jsonl

from utils.metrics import compute_cus, compute_zti


def run_gpt2():
    # Load data
    data = load_questions_from_jsonl("data/questions.jsonl")
    questions = [item["question"] for item in data][:10]  # Use only 10 for benchmark

    # Initialize model
    model = BayesianGPT2()
    tokenizer = model.tokenizer
    base_model = model.base_model
    model.eval()

    cus_no_uncertainty = []
    cus_with_uncertainty = []

    print("\n Running GPT2 Benchmark with Bayesian Inference...\n")

    for i, question in enumerate(tqdm(questions, desc="Running GPT2")):
        # CUS without uncertainty
        response = generate_response_no_uncertainty(base_model, tokenizer, question)
        inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            logits = base_model(**inputs).logits
        cus_no_u = compute_cus(logits.detach().numpy())
        cus_no_uncertainty.append(cus_no_u)

        # CUS with uncertainty (MC Dropout)
        _, uncertainty = estimate_uncertainty_gpt2(model, tokenizer, question)
        cus_u = compute_cus(uncertainty)
        cus_with_uncertainty.append(cus_u)

        # Print values in scientific notation
        print(f"[{i+1}] GPT2 CUS - NoU: {cus_no_u:.3e} | Uncertainty: {cus_u:.3e}")

    # ZTI computation
    zti_no_u = compute_zti(cus_no_uncertainty)
    zti_with_u = compute_zti(cus_with_uncertainty)

    # Save results
    results = {
        "Model": "GPT2",
        "CUS_No_Uncertainty": [float(x) for x in cus_no_uncertainty],
        "CUS_With_Uncertainty": [float(x) for x in cus_with_uncertainty],
        "ZTI_No_Uncertainty": float(zti_no_u),
        "ZTI_With_Uncertainty": float(zti_with_u),
    }

    save_path = "results/gpt2_result.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n Results saved to {save_path}")
