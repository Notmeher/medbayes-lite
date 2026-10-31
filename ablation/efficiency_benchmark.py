import torch
import torch.nn as nn
import numpy as np
import time
import json
import pandas as pd
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = "/home/casl/Desktop/Elias/Medbayes/test.json"
SAVE_DIR = "/home/casl/Desktop/Elias/Medbayes"
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

MC_SAMPLES = [5, 10, 20, 50]
ENSEMBLE_SIZE = 5
SWAG_SAMPLES = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# SAFE CUDA RESET (works on ALL PyTorch versions)
# ============================================================
def safe_cuda_reset():
    try:
        torch.cuda.reset_peak_memory_allocated()
    except:
        pass

# ============================================================
# LOAD DATA (your format)
# ============================================================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw["data"] if isinstance(raw, dict) and "data" in raw else raw

raw_data = load_json(DATA_PATH)

answer_map = {"A":0, "B":1, "C":2, "D":3, "E":4}

questions = []
contexts = []
labels = []

for item in raw_data:
    q = item.get("question", "")

    # Build context text
    ctx_list = item.get("context", [])
    ctx_text = ""
    for c in ctx_list:
        title = c.get("title", "")
        sentences = " ".join(c.get("sentences", []))
        ctx_text += f"{title} {sentences} "

    full_context = ctx_text.strip()

    ans = item.get("answer", "").strip()
    label = answer_map.get(ans, 0)

    questions.append(q)
    contexts.append(full_context)
    labels.append(label)

labels_np = np.array(labels)

# ============================================================
# PARAM COUNT
# ============================================================
def count_params(model):
    return sum(p.numel() for p in model.parameters())

# ============================================================
# BASELINE BENCHMARK
# ============================================================
def benchmark_model(model, tokenizer, samples=20):
    model.eval()
    model.to(DEVICE)

    safe_cuda_reset()
    latencies = []

    for i in range(samples):
        text = questions[i] + " " + contexts[i]
        inputs = tokenizer(text, return_tensors="pt",
                           truncation=True, max_length=512).to(DEVICE)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start.record()
        with torch.no_grad():
            _ = model(**inputs).logits
        end.record()
        torch.cuda.synchronize()

        latencies.append(start.elapsed_time(end))

    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
    lat_ms = float(np.mean(latencies))
    throughput = (1000.0 / lat_ms)

    return lat_ms, throughput, peak_mem

# ============================================================
# MEDBAYES-LITE (MC SAMPLING)
# ============================================================
class BayesianWrapper(nn.Module):
    def __init__(self, base_model, dropout_rate=0.3):
        super().__init__()
        self.base = base_model
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids, attention_mask):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        return self.dropout(out.logits)

def run_mc_sampling(model, tokenizer, text, n_samples=20):
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=512).to(DEVICE)
    preds = []
    model.train()

    for _ in range(n_samples):
        with torch.no_grad():
            logits = model(inputs["input_ids"], inputs["attention_mask"])
            probs = softmax(logits.squeeze().cpu().numpy(), axis=-1)
            preds.append(probs)

    return np.mean(preds, axis=0)

def benchmark_medbayes(wrapper, tokenizer, mc_samples):
    safe_cuda_reset()
    latencies = []

    for i in range(20):
        text = questions[i] + " " + contexts[i]

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start.record()
        _ = run_mc_sampling(wrapper, tokenizer, text, n_samples=mc_samples)
        end.record()
        torch.cuda.synchronize()

        latencies.append(start.elapsed_time(end))

    lat_ms = float(np.mean(latencies))
    throughput = (1000.0 / lat_ms)
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)

    return lat_ms, throughput, peak_mem

# ============================================================
# SWAG
# ============================================================
class SWAGModel(nn.Module):
    def __init__(self, base_model, swag_scale=0.03, n_samples=10):
        super().__init__()
        self.base = base_model
        self.scale = swag_scale
        self.n = n_samples

    def sample_weights(self):
        with torch.no_grad():
            for p in self.base.parameters():
                if p.requires_grad:
                    noise = torch.randn_like(p) * self.scale
                    p.add_(noise)

    def forward_pass(self, inputs):
        preds=[]
        for _ in range(self.n):
            self.sample_weights()
            with torch.no_grad():
                logits = self.base(**inputs).logits
                probs = softmax(logits.squeeze().cpu().numpy(), axis=-1)
            preds.append(probs)
        return np.mean(preds, axis=0)

def benchmark_swag(swag, tokenizer):
    safe_cuda_reset()
    latencies = []

    for i in range(10):
        text = questions[i] + " " + contexts[i]
        inputs = tokenizer(text, return_tensors="pt",
                           truncation=True, max_length=512).to(DEVICE)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start.record()
        _ = swag.forward_pass(inputs)
        end.record()
        torch.cuda.synchronize()

        latencies.append(start.elapsed_time(end))

    lat_ms = float(np.mean(latencies))
    throughput = (1000.0 / lat_ms)
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)

    return lat_ms, throughput, peak_mem

# ============================================================
# DEEP ENSEMBLE
# ============================================================
def benchmark_ensemble(tokenizer, n_models=5):
    safe_cuda_reset()

    models = [
        AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=5).to(DEVICE)
        for _ in range(n_models)
    ]

    latencies = []

    for i in range(10):
        text = questions[i] + " " + contexts[i]
        inputs = tokenizer(text, return_tensors="pt",
                           truncation=True, max_length=512).to(DEVICE)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start.record()
        for m in models:
            with torch.no_grad():
                _ = m(**inputs).logits
        end.record()
        torch.cuda.synchronize()

        latencies.append(start.elapsed_time(end))

    lat_ms = float(np.mean(latencies))
    throughput = (1000.0 / lat_ms)
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)

    params_total = sum(count_params(m) for m in models)

    return lat_ms, throughput, peak_mem, params_total

# ============================================================
# RUN EXPERIMENT
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=5)

results = []

# 1. Baseline
lat, th, mem = benchmark_model(base_model, tokenizer)
results.append(["ClinicalBERT (Baseline)", count_params(base_model),
                "-", f"{lat:.2f} ms", f"{th:.1f}/s", f"{mem:.2f} GB"])

# 2. MedBayes-Lite
for mc in MC_SAMPLES:
    wrapper = BayesianWrapper(base_model, dropout_rate=0.3).to(DEVICE)
    lat, th, mem = benchmark_medbayes(wrapper, tokenizer, mc)
    results.append([f"MedBayes-Lite (MC={mc})", count_params(base_model),
                    f"{mc} samples", f"{lat:.2f} ms", f"{th:.1f}/s", f"{mem:.2f} GB"])

# 3. SWAG
swag = SWAGModel(
    AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=5).to(DEVICE),
    swag_scale=0.03,
    n_samples=SWAG_SAMPLES
)
lat, th, mem = benchmark_swag(swag, tokenizer)
results.append([f"SWAG (n={SWAG_SAMPLES})", count_params(base_model),
                f"{SWAG_SAMPLES}", f"{lat:.2f} ms", f"{th:.1f}/s", f"{mem:.2f} GB"])

# 4. Deep Ensemble
lat, th, mem, params_total = benchmark_ensemble(tokenizer, n_models=ENSEMBLE_SIZE)
results.append([f"Deep Ensemble (N={ENSEMBLE_SIZE})", params_total,
                "-", f"{lat:.2f} ms", f"{th:.1f}/s", f"{mem:.2f} GB"])

# ============================================================
# SAVE RESULTS
# ============================================================
df = pd.DataFrame(results, columns=[
    "Model Variant", "Params", "MC/Ensemble", "Latency", "Throughput", "GPU Memory"
])

df.to_csv(f"{SAVE_DIR}/efficiency_results.csv", index=False)
df.to_json(f"{SAVE_DIR}/efficiency_results.json", indent=2)

print("\n=== EFFICIENCY BENCHMARK RESULTS ===\n")
print(df.to_string(index=False))

print("\nSaved results to:")
print(f"  {SAVE_DIR}/efficiency_results.csv")
print(f"  {SAVE_DIR}/efficiency_results.json")
