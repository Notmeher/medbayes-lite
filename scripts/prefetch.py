"""Prefetch datasets, fine-tuned encoder checkpoints, and the local generative model.

Runs standalone (no project imports) so it can execute in the background while the
rest of the package is built. Logs progress; failures are recorded but non-fatal so
one unavailable resource does not block the rest.
"""
import os
import sys
import time
import traceback

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(_ROOT, "results", "prefetch.log")
os.makedirs(os.path.dirname(LOG), exist_ok=True)


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def fetch_datasets():
    from datasets import load_dataset
    specs = [
        ("pubmedqa", lambda: load_dataset("qiaojin/PubMedQA", "pqa_labeled")),
        ("medmcqa", lambda: load_dataset("openlifescienceai/medmcqa")),
        ("medqa_usmle_4opt", lambda: load_dataset("GBaker/MedQA-USMLE-4-options")),
        ("mediq", lambda: load_dataset("stellalisy/mediq")),
    ]
    for name, fn in specs:
        try:
            ds = fn()
            sizes = {k: len(v) for k, v in ds.items()}
            log(f"DATASET ok {name}: {sizes}")
        except Exception as e:
            log(f"DATASET FAIL {name}: {e!r}")


def fetch_encoders():
    from huggingface_hub import snapshot_download
    repos = [
        "RUI525/BioBERT-finetune-MedMCQA-wo-context",
        "RUI525/ClinicalBERT-finetune-MedMCQA-wo-context",
        "RUI525/PubMedBERT-finetune-MedMCQA-wo-context",
        "blizrys/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext-finetuned-pubmedqa",
        # backbones for fine-tuning BERT-base baseline + tokenizers
        "bert-base-uncased",
        "dmis-lab/biobert-base-cased-v1.1",
        "emilyalsentzer/Bio_ClinicalBERT",
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    ]
    for r in repos:
        try:
            p = snapshot_download(r)
            log(f"ENCODER ok {r} -> {p}")
        except Exception as e:
            log(f"ENCODER FAIL {r}: {e!r}")


def fetch_generative():
    from huggingface_hub import snapshot_download
    model = os.environ.get("GEN_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    try:
        p = snapshot_download(model, ignore_patterns=["*.pth", "*.onnx"])
        log(f"GEN ok {model} -> {p}")
    except Exception as e:
        log(f"GEN FAIL {model}: {e!r}")


if __name__ == "__main__":
    stages = sys.argv[1:] or ["datasets", "encoders", "generative"]
    log(f"=== prefetch start: {stages} ===")
    if "datasets" in stages:
        fetch_datasets()
    if "encoders" in stages:
        fetch_encoders()
    if "generative" in stages:
        fetch_generative()
    log("=== prefetch done ===")
