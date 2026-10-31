import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm   

# Try Dirichlet Calibration (optional)
try:
    from dirichletcal.calib.fulldirichlet import FullDirichletCalibrator
    has_dirichlet = True
except ImportError:
    has_dirichlet = False
    print("DirichletCal not installed, skipping this method.")

# -------------------------
# Load MedQA dataset
# -------------------------
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

medqa_path = "/home/casl/Desktop/Elias/MedBayes/test.json"
raw_json = load_json(medqa_path)

if isinstance(raw_json, dict) and "data" in raw_json:
    medqa_data = raw_json["data"]
else:
    medqa_data = raw_json

answer_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
questions = [item["question"] for item in medqa_data]
labels = [answer_map[item["answer"]] for item in medqa_data]
choices = [item.get("options", ["A","B","C","D","E"]) for item in medqa_data]

# ------------------------------------------------
# MedBayes: BayesianBERT with MC Dropout
# ------------------------------------------------
class BayesianBERT(nn.Module):
    def __init__(self, base_model):
        super(BayesianBERT, self).__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        return self.dropout(outputs.logits)

def estimate_uncertainty(model, tokenizer, text, n_samples=20):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    preds = []
    model.train()
    for _ in range(n_samples):
        with torch.no_grad():
            logits = model(inputs["input_ids"], inputs["attention_mask"])
            probs = softmax(logits.squeeze().cpu().numpy(), axis=-1)
            preds.append(probs)
    preds = np.array(preds)
    mean_pred = preds.mean(axis=0)
    unc = preds.std(axis=0) / (mean_pred + 1e-6)
    return mean_pred, unc

# -----------------------------------------------------
# Metrics
# -----------------------------------------------------
def compute_ece(probs, labels, n_bins=15):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)
    ece = 0.0
    bins = np.linspace(0, 1, n_bins+1)
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i+1])
        if np.any(mask):
            ece += np.abs(np.mean(confidences[mask]) - np.mean(accuracies[mask])) * np.mean(mask)
    return ece

def compute_nll(logits, labels):
    return F.cross_entropy(torch.tensor(logits), torch.tensor(labels)).item()

def compute_cus(uncertainties):
    return float(np.mean(uncertainties))

def compute_zti(cus_scores, eps=1e-6):
    if len(cus_scores) < 2:
        return 1.0
    mean, std = np.mean(cus_scores), np.std(cus_scores)
    max_cus = max(np.max(cus_scores), eps)
    zti = 1 - (np.log1p(abs(mean - std)) / np.log1p(max_cus))
    return float(np.clip(zti, 0, 1))

# -------------------------------------------
# Load ClinicalBERT
# -------------------------------------------
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=5)
bayes_model = BayesianBERT(base_model)

# -------------------------------------------
# Collect logits + probs with progress bar
# -------------------------------------------
all_logits, all_probs = [], []
print("Collecting logits and probabilities...")
for q, opts in tqdm(zip(questions, choices), total=len(questions)):
    text = q + " " + " ".join(opts)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = base_model(**inputs).logits
    all_logits.append(logits.squeeze().cpu().numpy())
    all_probs.append(softmax(logits.squeeze().cpu().numpy(), axis=-1))

all_logits = np.array(all_logits)
all_probs = np.array(all_probs)
labels = np.array(labels)

# -------------------------------------------
# Post-hoc Calibration
# -------------------------------------------
results = []

print("Running post-hoc calibration...")
# MSP
results.append(("MSP",
    compute_ece(all_probs, labels),
    compute_nll(all_logits, labels),
    compute_cus(all_probs.std(axis=-1)),
    compute_zti([compute_cus(all_probs.std(axis=-1))])
))

# Temp Scaling
class TempScaler(nn.Module):
    def __init__(self): super().__init__(); self.temp = nn.Parameter(torch.ones(1))
    def forward(self, logits): return logits / self.temp

scaler = TempScaler()
crit = nn.CrossEntropyLoss()
optim_temp = optim.LBFGS([scaler.temp], lr=0.01, max_iter=50)
logits_t, labels_t = torch.tensor(all_logits, dtype=torch.float32), torch.tensor(labels)

def closure():
    optim_temp.zero_grad()
    loss = crit(scaler(logits_t), labels_t)
    loss.backward(); return loss

optim_temp.step(closure)
T_opt = scaler.temp.item()
probs_temp = softmax(all_logits / T_opt, axis=-1)

results.append(("Temperature Scaling",
    compute_ece(probs_temp, labels),
    compute_nll(all_logits / T_opt, labels),
    compute_cus(probs_temp.std(axis=-1)),
    compute_zti([compute_cus(probs_temp.std(axis=-1))])
))

# Isotonic Regression
conf_raw = np.max(all_probs, axis=1)
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(conf_raw, (labels == np.argmax(all_probs, axis=1)))
conf_iso = iso.predict(conf_raw)
probs_iso = np.copy(all_probs)
for i in range(len(probs_iso)):
    top_idx = np.argmax(probs_iso[i])
    probs_iso[i] *= conf_iso[i] / (probs_iso[i][top_idx] + 1e-12)

results.append(("Isotonic Regression",
    compute_ece(probs_iso, labels),
    compute_nll(np.log(probs_iso + 1e-12), labels),
    compute_cus(probs_iso.std(axis=-1)),
    compute_zti([compute_cus(probs_iso.std(axis=-1))])
))

# Dirichlet Calibration (if available)
if has_dirichlet:
    dir_cal = FullDirichletCalibrator()
    dir_cal.fit(all_logits, labels)
    probs_dir = dir_cal.predict_proba(all_logits)
    results.append(("Dirichlet Calibration",
        compute_ece(probs_dir, labels),
        compute_nll(np.log(probs_dir + 1e-12), labels),
        compute_cus(probs_dir.std(axis=-1)),
        compute_zti([compute_cus(probs_dir.std(axis=-1))])
    ))

# -----------------------------------------------
# MedBayes (ours, ClinicalBERT + MC Dropout)
# -----------------------------------------------
print("Running MedBayes uncertainty estimation...")
cus_list = []
for q, opts in tqdm(zip(questions, choices), total=len(questions)):
    text = q + " " + " ".join(opts)
    mean_pred, unc = estimate_uncertainty(bayes_model, tokenizer, text, n_samples=20)
    cus_list.append(compute_cus(unc))

results.append(("MedBayes (ours)",
    compute_ece(all_probs, labels),
    compute_nll(all_logits, labels),
    np.mean(cus_list),
    compute_zti(cus_list)
))

# ---------------------------------------------
# Final Table
# ---------------------------------------------
df = pd.DataFrame(results, columns=["Method", "ECE ↓", "NLL ↓", "CUS ↓", "ZTI ↑"])
print(df.to_string(index=False))
