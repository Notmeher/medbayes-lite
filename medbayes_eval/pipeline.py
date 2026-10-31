"""End-to-end encoder pipeline: fine-tune (or load) -> infer -> derive every UQ
method -> assign clinical buckets -> emit the per-sample record DataFrame.

A single forward pass set (deterministic + MC-dropout) is reused across methods.
Calibration-dependent methods (temperature, conformal) are fit on a held-out
calibration slice carved from the evaluation set, never from the test slice.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from . import config


def torch_manual_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
from .schema import Sample, make_record, records_to_frame
from .data.loaders import load_dataset_samples
from .buckets.rules import assign_buckets, sample_severity
from .models.finetune import finetune_backbone, TrainConfig
from .models.encoder import EncoderModel
from .models.encoding import is_mcq
from .models import uq_methods as uq

# how many eval items to reserve for calibration (temp scaling / conformal)
CALIB_SIZE = {"pubmedqa": 150, "medmcqa": 500, "medqa": 300, "mediq": 300, "mimic": 300}


def get_train_eval(dataset: str, max_eval: Optional[int] = None,
                   max_train: Optional[int] = None, load_train: bool = True):
    """Return (train_samples, calib_samples, test_samples) for a dataset."""
    if dataset == "pubmedqa":
        train = load_dataset_samples("pubmedqa", split="calibration") if load_train else []
        evals = load_dataset_samples("pubmedqa", split="test")
    elif dataset == "medmcqa":
        train = (load_dataset_samples("medmcqa", split="train", max_samples=max_train)
                 if load_train else [])
        evals = load_dataset_samples("medmcqa", split="validation")
    elif dataset == "medqa":
        train = (load_dataset_samples("medqa", split="train", max_samples=max_train)
                 if load_train else [])
        evals = load_dataset_samples("medqa", split="test")
    elif dataset == "mediq":
        train = load_dataset_samples("medqa", split="train", max_samples=max_train)
        evals = load_dataset_samples("mediq", split="test")
    else:
        raise KeyError(f"no train/eval recipe for dataset '{dataset}'")
    if max_eval:
        evals = evals[:max_eval]
    n_calib = min(CALIB_SIZE.get(dataset, 300), len(evals) // 3)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(evals))
    calib_idx = set(perm[:n_calib].tolist())
    calib = [evals[i] for i in range(len(evals)) if i in calib_idx]
    test = [evals[i] for i in range(len(evals)) if i not in calib_idx]
    return train, calib, test


def _bucket_annotations(samples: List[Sample]):
    buckets = [assign_buckets(s) for s in samples]
    sevs = [sample_severity(b) for b in buckets]
    return buckets, sevs


def _emit(samples, buckets, sevs, dataset, model_name, method, seed,
          probs, conf, unc):
    recs = []
    for i, s in enumerate(samples):
        recs.append(make_record(
            s, model=model_name, method=method, seed=seed,
            probs=probs[i], confidence=float(conf[i]), uncertainty=float(unc[i]),
            buckets=buckets[i], severity=sevs[i]))
    return recs


def _resolve_source(model_name, dataset, train, n_classes, seed, train_cfg):
    """Return (ckpt_or_hub, mcq, pretrained) for a model on a dataset."""
    spec = config.MODELS[model_name]
    if spec.kind == "encoder_finetune":
        ckpt = finetune_backbone(spec.name, dataset, train, n_classes, seed=seed,
                                 cfg=train_cfg)
        return ckpt, is_mcq(train[0]), False
    if spec.kind in ("encoder_mcq", "encoder_seqcls"):
        return spec.hf_id, (spec.kind == "encoder_mcq"), True
    raise ValueError(f"model '{model_name}' is not an encoder (kind={spec.kind})")


def run_encoder_dataset(
    dataset: str,
    model_name: str,
    *,
    seeds: Sequence[int] = (0,),
    methods: Sequence[str] = ("base", "temp_scaling", "mc_dropout", "medbayes_lite"),
    mc_samples: int = 20,
    mc_p: float = 0.2,
    train_cfg: Optional[TrainConfig] = None,
    max_eval: Optional[int] = None,
    max_train: Optional[int] = None,
    device: str = "cuda",
) -> pd.DataFrame:
    """Produce per-sample records for one (dataset, model) across seeds/methods.

    `model_name` is a key in config.MODELS; pretrained MCQ/seqcls checkpoints are
    loaded directly (strong, real predictions) while encoder_finetune models are
    trained with the shared routine. Deterministic methods are identical across
    seeds for pretrained models (variance comes from MC sampling).
    """
    train, calib, test = get_train_eval(dataset, max_eval=max_eval, max_train=max_train)
    n_classes = test[0].n_classes
    mcq = is_mcq(test[0])
    buckets, sevs = _bucket_annotations(test)
    train_cfg = train_cfg or TrainConfig(max_train=max_train)

    need_mc = any(m in methods for m in
                  ("mc_dropout", "medbayes_lite", "mc_only", "uncertainty_attention"))
    all_records: List[dict] = []
    per_seed_test_logits = []

    for seed in seeds:
        torch_manual_seed(seed)
        ckpt, mcq, pretrained = _resolve_source(
            model_name, dataset, train, n_classes, seed, train_cfg)
        mlen = train_cfg.max_len_mcq if mcq else train_cfg.max_len_seq
        model = EncoderModel(ckpt, mcq=mcq, device=device, max_len=mlen)

        test_logits = model.logits(test)
        calib_logits = model.logits(calib)
        calib_labels = np.array([s.label for s in calib])
        per_seed_test_logits.append(test_logits)

        mc_logits = model.mc_logits(test, mc_samples, mc_p=mc_p) if need_mc else None
        T = uq.fit_temperature(calib_logits, calib_labels) if "temp_scaling" in methods else 1.0
        # MedBayes-Lite calibrates the MC-averaged distribution (not single-pass
        # logits) to avoid double-softening; fit its temperature on the calib BMA.
        T_mbl = 1.0
        if "medbayes_lite" in methods:
            calib_mc = model.mc_logits(calib, mc_samples, mc_p=mc_p)
            calib_bma = uq.softmax(calib_mc, axis=-1).mean(axis=0)
            T_mbl = uq.fit_temperature_on_probs(calib_bma, calib_labels)

        method_outputs = {}
        if "base" in methods or "msp" in methods:
            method_outputs["base"] = uq.predict_base(test_logits)
        if "temp_scaling" in methods:
            method_outputs["temp_scaling"] = uq.predict_temperature(test_logits, T)
        if "conformal" in methods:
            method_outputs["conformal"] = uq.predict_conformal(
                test_logits, calib_logits, calib_labels)
        if "abstain_only" in methods:
            method_outputs["abstain_only"] = uq.predict_abstain_only(test_logits)
        if need_mc:
            if "mc_dropout" in methods:
                method_outputs["mc_dropout"] = uq.predict_mc_dropout(mc_logits)
            if "mc_only" in methods:
                method_outputs["mc_only"] = uq.predict_mc_dropout(mc_logits)
            if "medbayes_lite" in methods:
                method_outputs["medbayes_lite"] = uq.predict_medbayes_lite(mc_logits, T_mbl)
            if "uncertainty_attention" in methods:
                method_outputs["uncertainty_attention"] = \
                    uq.predict_uncertainty_attention(mc_logits)

        for method, (probs, conf, unc) in method_outputs.items():
            all_records += _emit(test, buckets, sevs, dataset, model_name, method,
                                  seed, probs, conf, unc)
        del model

    # deep ensemble across separately fine-tuned per-seed checkpoints (genuine only
    # when the checkpoints differ, i.e. encoder_finetune models with >1 seed).
    spec = config.MODELS[model_name]
    if ("deep_ensemble" in methods and spec.kind == "encoder_finetune"
            and len(per_seed_test_logits) > 1):
        probs, conf, unc = uq.predict_deep_ensemble(per_seed_test_logits)
        all_records += _emit(test, buckets, sevs, dataset, model_name, "deep_ensemble",
                             -1, probs, conf, unc)

    return records_to_frame(all_records)


def save_predictions(df: pd.DataFrame, dataset: str, model_name: str,
                     results_dir: str = config.RESULTS_DIR):
    out = os.path.join(results_dir, "raw_predictions", f"{dataset}__{model_name}.parquet")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_parquet(out, index=False)
    return out
