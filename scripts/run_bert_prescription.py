# scripts/run_bert_prescription.py

import os
import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from models.bert_prescription_model import BayesianBERT
from utils.metrics import compute_cus, compute_zti

def run_bert_prescription():
    df = pd.read_csv("data/PRESCRIPTIONS.csv", low_memory=False)
    questions = df["DRUG_NAME_GENERIC"].dropna().astype(str).tolist()[:10]

    BERT_MODELS = {
        "BERT-Base": "bert-base-uncased",
        "BERT-Large": "bert-large-uncased",
        "BioBERT": "dmis-lab/biobert-base-cased-v1.1",
        "ClinicalBERT": "emilyalsentzer/Bio_ClinicalBERT",
        "RoBERTa": "roberta-base"
    }

    results = {}

    for name, model_id in BERT_MODELS.items():
        print(f"\n====== Running {name} ======")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=4, ignore_mismatched_sizes=True
        )
        bayesian_model = BayesianBERT(base_model)

        no_unc = []
        with_unc = []

        for i, q in enumerate(questions):
            input = tokenizer(q, return_tensors="pt", padding=True, truncation=True, max_length=512)
            if "token_type_ids" not in input:
                input["token_type_ids"] = None

            # No uncertainty
            with torch.no_grad():
                out = base_model(**input).logits
            probs = torch.softmax(out, dim=1).numpy()
            no_unc.append(float(torch.tensor(probs).mean()))

            # With uncertainty
            predictions = []
            for _ in range(10):
                with torch.no_grad():
                    pred = bayesian_model(**input)
                    predictions.append(pred.numpy())
            predictions = torch.tensor(predictions)
            std = predictions.std(dim=0).mean().item()
            with_unc.append(std)

            print(f"[{i+1}] CUS (with uncertainty): {std:.4f}")

        zti_no = compute_zti(no_unc)
        zti_with = compute_zti(with_unc)

        print(f"{name} ZTI (No Uncertainty):", zti_no)
        print(f"{name} ZTI (With Uncertainty):", zti_with)

        results[name] = {
            "CUS_No_Uncertainty": no_unc,
            "CUS_With_Uncertainty": with_unc,
            "ZTI_No_Uncertainty": zti_no,
            "ZTI_With_Uncertainty": zti_with
        }

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/bert_prescription_result.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_bert_prescription()
