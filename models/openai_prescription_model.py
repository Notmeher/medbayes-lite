# models/openai_prescription_model.py

import os
import pandas as pd
import json
from dotenv import load_dotenv
from models.openai_prescription_model import OpenAIPrescriptionModel
from utils.metrics import compute_cus, compute_zti

# Load API key from .env
load_dotenv()

def run_openai_prescription():
    df = pd.read_csv("data/PRESCRIPTIONS.csv", low_memory=False)
    questions = df["DRUG_NAME_GENERIC"].dropna().astype(str).tolist()[:10]

    MODELS = {
        "GPT-4.5": "gpt-4.5-preview",
        "GPT-4o": "gpt-4o",
        "GPT-4": "gpt-4",
        "GPT-3.5": "gpt-3.5-turbo"
    }

    for name, model_key in MODELS.items():
        print(f"\n====== Running {name} ======")
        model = OpenAIPrescriptionModel(model_name=model_key)

        no_unc = []
        with_unc = []
        full_responses = []

        for i, q in enumerate(questions):
            resp = model.generate_response(q)
            print(f"[{i+1}] Response (no uncertainty): {resp}")
            no_unc.append(len(resp.split()))
            full_responses.append(resp)

            _, unc = model.estimate_uncertainty(q)
            with_unc.append(unc)
            print(f"[{i+1}] CUS (with uncertainty): {unc:.4f}")

        zti_no = compute_zti(no_unc)
        zti_unc = compute_zti(with_unc)

        print(f"{name} ZTI (No Uncertainty):", zti_no)
        print(f"{name} ZTI (With Uncertainty):", zti_unc)

        # Save to results
        result_dict = {
            "model": name,
            "responses": full_responses,
            "response_lengths": no_unc,
            "CUS": with_unc,
            "ZTI_no_uncertainty": zti_no,
            "ZTI_with_uncertainty": zti_unc
        }

        os.makedirs("results", exist_ok=True)
        output_path = os.path.join("results", f"openai_{name.lower().replace('.', '').replace('-', '')}_result.json")
        with open(output_path, "w") as f:
            json.dump(result_dict, f, indent=4)

        print(f"[✓] Results saved to {output_path}")
