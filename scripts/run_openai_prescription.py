# scripts/run_openai_prescription.py

import os
import pandas as pd
from models.openai_prescription_model import OpenAIPrescriptionModel
from utils.metrics import compute_cus, compute_zti
from dotenv import load_dotenv

# Load API Key from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(" OPENAI_API_KEY not found in environment. Check .env or terminal `export`.")

print(f"[INFO] Loaded API key: {api_key[:5]}...***")  # Masked for safety

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

        # Instantiate with API key explicitly (optional, since the model handles env too)
        model = OpenAIPrescriptionModel(model_name=model_key, api_key=api_key)

        no_unc = []
        with_unc = []

        for i, q in enumerate(questions):
            resp = model.generate_response(q)
            print(f"[{i+1}] Response (no uncertainty): {resp}")
            no_unc.append(len(resp.split()))

            _, unc = model.estimate_uncertainty(q)
            with_unc.append(unc)
            print(f"[{i+1}] CUS (with uncertainty): {unc:.4f}")

        print(f"{name} ZTI (No Uncertainty):", compute_zti(no_unc))
        print(f"{name} ZTI (With Uncertainty):", compute_zti(with_unc))
