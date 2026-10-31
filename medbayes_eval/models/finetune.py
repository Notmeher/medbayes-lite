"""Unified fine-tuning of encoder backbones, identical routine across all model
families so calibration/uncertainty differences are attributable to the backbone,
not the training protocol.

Supports multiple-choice heads (MedQA/MedMCQA/MediQ) and sequence-classification
heads (PubMedQA 3-way, MIMIC binary). Checkpoints are cached under
results/checkpoints/<dataset>/<backbone>/seed<k>/ and reused on subsequent runs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_linear_schedule_with_warmup
from .. import config
from ..schema import Sample
from .encoding import build_mcq_features, build_seqcls_features, labels_tensor, is_mcq
from .encoder_fast import load_encoder, load_tokenizer

# canonical backbone hub ids for the four encoder families
BACKBONES = {
    "biobert": "dmis-lab/biobert-base-cased-v1.1",
    "clinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
    "pubmedbert": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "bert_base": "bert-base-uncased",
}

CKPT_ROOT = config.CKPT_DIR


@dataclass
class TrainConfig:
    epochs: int = 2
    lr: float = 3e-5
    batch_size: int = 16
    max_len_mcq: int = 320
    max_len_seq: int = 512
    max_train: Optional[int] = None     # cap training set (None = full)
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    fp16: bool = True
    device: str = "cuda"


def ckpt_dir(dataset: str, backbone: str, seed: int) -> str:
    return os.path.join(CKPT_ROOT, dataset, backbone, f"seed{seed}")


def _set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_features(tokenizer, samples, mcq, cfg):
    if mcq:
        feats, _ = build_mcq_features(tokenizer, samples, cfg.max_len_mcq)
    else:
        feats = build_seqcls_features(tokenizer, samples, cfg.max_len_seq)
    return feats


def finetune_backbone(backbone: str, dataset: str, train_samples: List[Sample],
                      n_classes: int, seed: int = 0, cfg: Optional[TrainConfig] = None,
                      force: bool = False) -> str:
    """Fine-tune one backbone on one dataset; return the checkpoint directory.

    Reuses an existing checkpoint unless force=True.
    """
    cfg = cfg or TrainConfig()
    out_dir = ckpt_dir(dataset, backbone, seed)
    if os.path.exists(os.path.join(out_dir, "config.json")) and not force:
        return out_dir

    _set_seed(seed)
    mcq = is_mcq(train_samples[0])
    hub = BACKBONES[backbone]
    tokenizer = load_tokenizer(hub)
    model = load_encoder(hub, mcq, num_labels=None if mcq else n_classes)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    model.to(device)

    samples = train_samples
    if cfg.max_train:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(samples))[: cfg.max_train]
        samples = [samples[i] for i in idx]

    feats = _build_features(tokenizer, samples, mcq, cfg)
    labels = labels_tensor(samples)
    tensors = list(feats.values()) + [labels]
    keys = list(feats.keys())
    ds = TensorDataset(*tensors)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = len(loader) * cfg.epochs
    sched = get_linear_schedule_with_warmup(
        optim, int(total_steps * cfg.warmup_ratio), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.fp16 and device == "cuda")

    model.train()
    for epoch in range(cfg.epochs):
        running = 0.0
        for batch in loader:
            *inputs, y = batch
            batch_in = {k: v.to(device) for k, v in zip(keys, inputs)}
            y = y.to(device)
            optim.zero_grad()
            with torch.autocast("cuda", enabled=cfg.fp16 and device == "cuda"):
                out = model(**batch_in, labels=y)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            sched.step()
            running += loss.item()
        print(f"  [{backbone}/{dataset}/seed{seed}] epoch {epoch+1}/{cfg.epochs} "
              f"loss={running/len(loader):.4f}", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return out_dir
