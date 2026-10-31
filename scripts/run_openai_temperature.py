# scripts/run_openai_temperature.py

import os
import json
import numpy as np
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

from utils.metrics import compute_confidence_entropy, compute_zti

# Load environment variables
load_dotenv("configs/.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load data
df = pd.read_csv("data/PRESCRIPTIONS.csv", low_memory=False)
questions = df["DRUG_NAME_GENERIC"].dropna().astype(str).tolist()
num_samples = min(10, len(questions))

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Temperatures to test
TEMPERATURES = [1.0, 0.8, 1.2]

# Initialize result container
all_results = {}

def generate_response(client, _, input_text, temperature=1.0):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": input_text}],
        max_tokens=50,
        temperature=temperature
    )
    return response.choices[0].message.content

def run_openai_temperature():
    print(" Evaluating GPT-3.5 with Temperature Scaling...\n")

    for temp in TEMPERATURES:
        print(f"\n Temperature = {temp}")
        confidence_entropies = []
        cus_scores = []

        for i in tqdm(range(num_samples), desc=f"Temp {temp}"):
            question = questions[i]
            response = generate_response(client, None, question, temperature=temp)
            print(f"[{i+1}] {question} → {response.strip().split('.')[0]}")

            # Cast to float to avoid type errors during division
            probs = np.array([len(response.split())], dtype=np.float64)
            probs /= np.sum(probs)

            entropy = compute_confidence_entropy(probs)
            confidence_entropies.append(entropy)
            cus_scores.append(np.mean(probs))

        zti = compute_zti(cus_scores)

        # Save results per temperature
        all_results[f"GPT-3.5-T{temp}"] = {
            "CUS": float(np.mean(cus_scores)),
            "ZTI": float(zti),
            "Confidence_Entropy": float(np.mean(confidence_entropies))
        }

    # Save to JSON
    os.makedirs("results", exist_ok=True)
    result_path = os.path.join("results", "openai_prescription_results.json")
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n Results saved to {result_path}")
