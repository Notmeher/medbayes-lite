"""GPU jobs. Each job writes one atomic cache file and is skipped if it exists, so an
interrupted Kaggle session resumes where it stopped."""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Optional, Sequence

import numpy as np

from .. import config
from ..models.encoding import is_mcq
from ..pipeline import get_train_eval
from . import paths

MC_SEED_OFFSET = 10_000      # main caches
PERTURB_SEED_OFFSET = 20_000
SENS_SEED_OFFSET = 30_000


def _save_npz(path: str, meta: dict, **arrays):
    tmp = path[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta, default=str)), **arrays)
    os.replace(tmp, path)


def load_npz(path: str):
    d = np.load(path, allow_pickle=False)
    arrays = {k: d[k] for k in d.files if k != "meta"}
    return json.loads(str(d["meta"])), arrays


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ckpt_hash(ckpt: str) -> Optional[str]:
    for name in ("model.safetensors", "pytorch_model.bin"):
        p = os.path.join(ckpt, name)
        if os.path.exists(p):
            return sha256_file(p)
    return None


def _resolve(model: str, dataset: str, train, n_classes: int, seed: int,
             max_train: Optional[int]):
    """Return (checkpoint, provenance dict). Fine-tunes PubMedQA encoders on demand."""
    spec = config.MODELS[model]
    if spec.kind == "encoder_finetune":
        from ..models.finetune import TrainConfig, finetune_backbone
        cfg = TrainConfig(max_train=max_train)
        ckpt = finetune_backbone(spec.name, dataset, train, n_classes, seed=seed, cfg=cfg)
        return ckpt, {"kind": "finetuned", "backbone": spec.hf_id,
                      "backbone_revision": config.model_revision(spec.hf_id),
                      "ckpt_sha256": _ckpt_hash(ckpt), "train_config": vars(cfg),
                      "n_train": len(train)}
    return spec.hf_id, {"kind": "pretrained", "hf_id": spec.hf_id,
                        "revision": config.model_revision(spec.hf_id)}


def _encoder(ckpt: str, mcq: bool, batch_size: int):
    from ..models.encoder_fast import FastEncoder
    return FastEncoder(ckpt, mcq=mcq, max_len=320 if mcq else 512, batch_size=batch_size)


def _gpu_name():
    import torch
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
def encoder_logits(dataset: str, model: str, seed: int, mc_samples: int = 15,
                   mc_p: float = 0.2, max_eval: Optional[int] = None,
                   max_train: Optional[int] = None, batch_size: int = 64):
    """Deterministic + MC-dropout logits on the calibration and test slices."""
    out = paths.logits_path(dataset, model, seed)
    if os.path.exists(out):
        return out
    finetune = config.MODELS[model].kind == "encoder_finetune"
    train, calib, test = get_train_eval(dataset, max_eval=max_eval, max_train=max_train,
                                        load_train=finetune)
    ckpt, prov = _resolve(model, dataset, train, test[0].n_classes, seed, max_train)
    enc = _encoder(ckpt, is_mcq(test[0]), batch_size)
    pt, pc = enc.prepare(test), enc.prepare(calib)
    test_det, t_det = enc.timed(enc.logits, pt)
    calib_det = enc.logits(pc)
    test_mc, t_mc = enc.timed(enc.mc_logits, pt, mc_samples, mc_p, MC_SEED_OFFSET + seed)
    calib_mc = enc.mc_logits(pc, mc_samples, mc_p, MC_SEED_OFFSET + 1000 + seed)
    meta = {"dataset": dataset, "model": model, "seed": seed, "mc_samples": mc_samples,
            "mc_p": mc_p, "mc_seed": MC_SEED_OFFSET + seed, "ckpt": ckpt, **prov,
            "max_len": enc.max_len, "n_test": len(test), "n_calib": len(calib),
            "n_classes": test[0].n_classes, "gpu": _gpu_name(),
            "seconds_det_test": t_det, "seconds_mc_test": t_mc}
    _save_npz(out, meta,
              test_uids=np.array([s.uid for s in test]),
              calib_uids=np.array([s.uid for s in calib]),
              test_labels=np.array([s.label for s in test]),
              calib_labels=np.array([s.label for s in calib]),
              test_det=test_det, calib_det=calib_det,
              test_mc=test_mc.astype(np.float32), calib_mc=calib_mc.astype(np.float32))
    return out


# ---------------------------------------------------------------------------
def sensitivity(dataset: str = "medmcqa", model: str = "pubmedbert_medmcqa",
                max_m: int = 50, p_main: float = 0.2,
                p_grid: Sequence[float] = (0.1, 0.3, 0.5), m_grid_p: int = 15,
                max_eval: Optional[int] = None, batch_size: int = 64):
    """MC-sample sweep (one M=max_m stack, subsampled) and dropout-rate sweep."""
    out = os.path.join(paths.SENS, f"{dataset}__{model}.npz")
    if os.path.exists(out):
        return out
    _, calib, test = get_train_eval(dataset, max_eval=max_eval, load_train=False)
    enc = _encoder(config.MODELS[model].hf_id, is_mcq(test[0]), batch_size)
    pt, pc = enc.prepare(test), enc.prepare(calib)
    arrays = {"test_labels": np.array([s.label for s in test]),
              "calib_labels": np.array([s.label for s in calib]),
              "test_uids": np.array([s.uid for s in test]),
              "test_det": enc.logits(pt), "calib_det": enc.logits(pc)}
    for j, (p, m) in enumerate([(p_main, max_m)] + [(p, m_grid_p) for p in p_grid]):
        tag = f"p{p:g}"
        arrays[f"test_mc_{tag}"] = enc.mc_logits(pt, m, p, SENS_SEED_OFFSET + 2 * j)
        arrays[f"calib_mc_{tag}"] = enc.mc_logits(pc, m, p, SENS_SEED_OFFSET + 2 * j + 1)
    meta = {"dataset": dataset, "model": model, "max_m": max_m, "p_main": p_main,
            "p_grid": list(p_grid), "m_grid_p": m_grid_p, "gpu": _gpu_name()}
    _save_npz(out, meta, **arrays)
    return out


# ---------------------------------------------------------------------------
def perturb(dataset: str, model: str = "pubmedbert_medmcqa", n: int = 300,
            seeds: Sequence[int] = (0, 1, 2), mc_samples: int = 15, mc_p: float = 0.2,
            batch_size: int = 64):
    """Clean vs perturbed logits (det + seeded MC) for every perturbation."""
    from ..perturb.perturbations import PERTURBATIONS, apply_perturbation
    out = os.path.join(paths.PERTURB, f"{dataset}__{model}.npz")
    if os.path.exists(out):
        return out
    _, calib, test = get_train_eval(dataset, load_train=False)
    # same subset and rng stream as the original `run.py perturb`
    rng = np.random.default_rng(0)
    test = [test[i] for i in rng.permutation(len(test))[:n]]
    conditions = {"clean": test}
    for name in PERTURBATIONS:
        conditions[name] = [apply_perturbation(s, name, rng) for s in test]
    enc = _encoder(config.MODELS[model].hf_id, is_mcq(test[0]), batch_size)
    pc = enc.prepare(calib)
    arrays = {"labels": np.array([s.label for s in test]),
              "uids": np.array([s.uid for s in test]),
              "calib_labels": np.array([s.label for s in calib]),
              "calib_det": enc.logits(pc)}
    for sd in seeds:
        arrays[f"calib_mc{sd}"] = enc.mc_logits(pc, mc_samples, mc_p,
                                               PERTURB_SEED_OFFSET + 1000 + sd)
    texts = {}
    for cond, samples in conditions.items():
        pp = enc.prepare(samples)
        arrays[f"{cond}__det"] = enc.logits(pp)
        arrays[f"{cond}__applied"] = np.array(
            [cond == "clean" or bool(s.meta.get("perturb_applied")) for s in samples])
        for sd in seeds:
            arrays[f"{cond}__mc{sd}"] = enc.mc_logits(pp, mc_samples, mc_p,
                                                     PERTURB_SEED_OFFSET + sd)
        texts[cond] = [s.display_text() for s in samples]
    meta = {"dataset": dataset, "model": model, "n": n, "seeds": list(seeds),
            "mc_samples": mc_samples, "mc_p": mc_p, "conditions": list(conditions),
            "texts": texts, "gpu": _gpu_name()}
    _save_npz(out, meta, **arrays)
    return out


# ---------------------------------------------------------------------------
def generative(datasets: Sequence[str] = ("pubmedqa", "medmcqa", "medqa"), n: int = 200,
               n_samples: int = 10, temperature: float = 0.7, batch_size: int = 8,
               seed: int = 0, model_id: str = "Qwen/Qwen2.5-7B-Instruct",
               device_map: str = "auto", chunk: int = 40):
    """Qwen2.5-7B sampling-only UQ (both GPUs via device_map='auto'). Checkpoints
    every `chunk` items so a restart resumes mid-dataset."""
    import pandas as pd
    import torch
    from ..buckets.rules import assign_buckets, sample_severity
    from ..data.loaders import load_dataset_samples
    from ..models.generative import GenerativeModel
    from ..schema import make_record, records_to_frame

    model = None
    outs = []
    for ds in datasets:
        final = os.path.join(paths.GEN, f"{ds}__qwen2_5_7b.parquet")
        if os.path.exists(final):
            outs.append(final)
            continue
        split = "validation" if ds == "medmcqa" else "test"
        samples = load_dataset_samples(ds, split=split, max_samples=n)
        parts = []
        for c0 in range(0, len(samples), chunk):
            part = os.path.join(paths.GEN, f"{ds}__part{c0:05d}.parquet")
            if not os.path.exists(part):
                if model is None:
                    model = GenerativeModel(model_id, dtype=torch.float16,
                                            device_map=device_map)
                recs, raw = [], []
                t0 = time.time()
                block = samples[c0:c0 + chunk]
                for b0 in range(0, len(block), batch_size):
                    batch = block[b0:b0 + batch_size]
                    res = model.predict_shared_batch(
                        batch, n_samples=n_samples, temperature=temperature,
                        seed=seed * 1_000_003 + c0 + b0)
                    for s, r in zip(batch, res):
                        b = assign_buckets(s)
                        for method in ("self_consistency", "semantic_entropy", "verbalized"):
                            probs, conf, unc = r[method]
                            recs.append(make_record(s, model="qwen2_5_7b", method=method,
                                                    seed=seed, probs=probs, confidence=conf,
                                                    uncertainty=unc, buckets=b,
                                                    severity=sample_severity(b)))
                        raw.append({"uid": s.uid, **r["_raw"]})
                df = records_to_frame(recs)
                df.to_parquet(part[:-8] + ".tmp.parquet", index=False)
                os.replace(part[:-8] + ".tmp.parquet", part)
                with open(part[:-8] + ".raw.jsonl", "w") as f:
                    for row in raw:
                        f.write(json.dumps(row) + "\n")
                print(f"[gen] {ds} items {c0}-{c0 + len(block)} in {time.time() - t0:.0f}s",
                      flush=True)
            parts.append(part)
        pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True).to_parquet(
            final, index=False)
        outs.append(final)
    return outs


JOB_KINDS = {"encoder_logits": encoder_logits, "sensitivity": sensitivity,
             "perturb": perturb, "generative": generative}


def run_job(job: dict):
    from .latency import latency_benchmark
    kinds = {**JOB_KINDS, "latency": latency_benchmark}
    return kinds[job["kind"]](**job.get("args", {}))
