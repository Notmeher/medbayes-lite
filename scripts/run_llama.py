# scripts/run_llama.py

import json
import torch
from tqdm import tqdm
from models.llama_model import BayesianLLaMA, generate_response_no_uncertainty, estimate_uncertainty_llama
from utils.data_loader import load_questions_from_jsonl
from utils.metrics import compute_cus, compute_zti

def run_llama():
    data_path = "data/questions.jsonl"
    questions = load_questions_from_jsonl(data_path)
    num_samples = min(10, len(questions))

    model = BayesianLLaMA()
    tokenizer = model.tokenizer
    base_model = model.base_model

    cus_no_uncertainty = []
    cus_with_uncertainty = []

    print("\n🦙 Running LLaMA Benchmark with Bayesian Inference...\n")

    for i in tqdm(range(num_samples), desc="Running LLaMA"):
        question = str(questions[i])

        # Without uncertainty
        response = generate_response_no_uncertainty(base_model, tokenizer, question)
        inputs = tokenizer(question, return_tensors="pt")
        with torch.no_grad():
            logits = base_model(**inputs).logits
        cus_no_u = compute_cus(logits.detach().numpy())
        cus_no_uncertainty.append(cus_no_u)

        # With uncertainty
        _, uncertainty = estimate_uncertainty_llama(model, tokenizer, question)
        cus_u = compute_cus(uncertainty)
        cus_with_uncertainty.append(cus_u)

        print(f"[{i+1}] LLaMA CUS - NoU: {cus_no_u:.3e} | Uncertainty: {cus_u:.3e}")

    zti_no_u = compute_zti(cus_no_uncertainty)
    zti_with_u = compute_zti(cus_with_uncertainty)

    results = {
        "Model": "LLaMA",
        "CUS_No_Uncertainty": [float(x) for x in cus_no_uncertainty],
        "CUS_With_Uncertainty": [float(x) for x in cus_with_uncertainty],
        "ZTI_No_Uncertainty": float(zti_no_u),
        "ZTI_With_Uncertainty": float(zti_with_u),
    }

    save_path = "results/llama_result.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to {save_path}")
