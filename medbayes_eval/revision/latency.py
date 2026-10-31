"""Latency / throughput / memory benchmark (R3:2 matched latency, R3:9 compute cost).

Replaces ablation/efficiency_benchmark.py, whose MC wrapper only applied dropout to
the output logits and whose baseline timing included CUDA warm-up. Here MC-dropout
runs real stochastic transformer passes, every configuration is warmed up, timed
with CUDA synchronization, and peak memory is reset per configuration.
"""
from __future__ import annotations

import copy
import time
from typing import Sequence

import numpy as np
import pandas as pd

from .. import config
from ..models.encoding import is_mcq
from ..pipeline import get_train_eval
from . import paths


def _sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time(fn, reps: int, warmup: int = 3):
    for _ in range(warmup):
        fn()
    _sync()
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        _sync()
        t.append(time.perf_counter() - t0)
    return float(np.median(t)), float(np.percentile(t, 90))


def _bench_backbone(label, ckpt, mcq, samples, mc_grid, ens_sizes, swag_n, reps,
                    batch_size):
    import torch
    from ..models.encoder_fast import FastEncoder, load_encoder
    from ..models.encoder import _enable_mc_dropout
    enc = FastEncoder(ckpt, mcq=mcq, max_len=320 if mcq else 512, batch_size=batch_size)
    dev = enc.device
    single = enc.prepare(samples[:1])
    batched = enc.prepare(samples[:batch_size])
    n_params = sum(p.numel() for p in enc.model.parameters())
    rows = []

    def record(method, passes, fn_item, fn_batch, params=n_params, extra=None):
        if dev == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        lat_med, lat_p90 = _time(fn_item, reps)
        bt, _ = _time(fn_batch, max(3, reps // 5))
        mem = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else float("nan")
        rows.append({"backbone": label, "method": method, "forward_passes": passes,
                     "params": params, "latency_ms_bs1_median": 1000 * lat_med,
                     "latency_ms_bs1_p90": 1000 * lat_p90,
                     "throughput_items_per_s": batch_size / bt,
                     "peak_gpu_mem_gb": mem, **(extra or {})})

    T = 1.5  # any constant: cost of the temperature step is what is measured
    record("base (single pass)", 1, lambda: enc.logits(single), lambda: enc.logits(batched))
    record("temperature scaling", 1,
           lambda: torch.softmax(torch.from_numpy(enc.logits(single)) / T, -1),
           lambda: torch.softmax(torch.from_numpy(enc.logits(batched)) / T, -1))
    for m in mc_grid:
        record(f"MC dropout / MedBayes-Lite (M={m})", m,
               lambda m=m: enc.mc_logits(single, m, 0.2, 0),
               lambda m=m: enc.mc_logits(batched, m, 0.2, 0), extra={"M": m})
    # MC with the M passes packed into one batch (same cost model as a GPU-parallel MC)
    for m in [x for x in mc_grid if x <= 32]:
        rep = enc.prepare(samples[:1] * m)

        def packed(rep=rep):
            _enable_mc_dropout(enc.model, p=0.2)
            return enc.logits(rep, mc_dropout=True, mc_p=0.2)
        record(f"MC dropout packed batch (M={m})", m, packed, packed, extra={"M": m})
    enc.model.eval()

    for n_ens in ens_sizes:
        members = [enc.model] + [copy.deepcopy(enc.model) for _ in range(n_ens - 1)]

        def run(prep, members=members):
            out = []
            for mdl in members:
                enc.model = mdl
                out.append(enc.logits(prep))
            enc.model = members[0]
            return out
        record(f"deep ensemble (N={n_ens})", n_ens, lambda: run(single), lambda: run(batched),
               params=n_params * n_ens, extra={"N": n_ens})
        del members
        if dev == "cuda":
            torch.cuda.empty_cache()

    if swag_n:
        base_state = {k: v.clone() for k, v in enc.model.state_dict().items()}
        std = {k: 0.01 * v.abs().mean().clamp(min=1e-6) for k, v in base_state.items()
               if v.dtype.is_floating_point}

        def swag(prep):
            out = []
            sd = enc.model.state_dict()
            for _ in range(swag_n):
                with torch.no_grad():
                    for k, s in std.items():
                        sd[k].copy_(base_state[k] + s * torch.randn_like(base_state[k]))
                out.append(enc.logits(prep))
            enc.model.load_state_dict(base_state)
            return out
        record(f"SWAG cost proxy (n={swag_n} weight samples)", swag_n,
               lambda: swag(single), lambda: swag(batched),
               extra={"note": "diagonal weight sampling; cost only, not a fitted SWAG"})
    return rows


def latency_benchmark(mc_grid: Sequence[int] = (1, 2, 3, 5, 10, 15, 20, 30, 50),
                      ens_sizes: Sequence[int] = (3, 5), swag_n: int = 10, reps: int = 30,
                      batch_size: int = 32, n_items: int = 64):
    out = paths.out("tables", "latency_benchmark.csv")
    _, _, mcq_test = get_train_eval("medmcqa", max_eval=600, load_train=False)
    _, _, pq_test = get_train_eval("pubmedqa", load_train=False)
    rows = []
    rows += _bench_backbone("PubMedBERT (MedMCQA, 4-way MC)",
                            config.MODELS["pubmedbert_medmcqa"].hf_id, True,
                            mcq_test[:n_items], mc_grid, ens_sizes, swag_n, reps, batch_size)
    rows += _bench_backbone("Bio_ClinicalBERT (PubMedQA, 3-way, len 512)",
                            "emilyalsentzer/Bio_ClinicalBERT", False,
                            pq_test[:n_items], mc_grid, ens_sizes, swag_n, reps, batch_size)
    df = pd.DataFrame(rows)
    from .jobs import _gpu_name
    df["gpu"] = _gpu_name()
    df.to_csv(out, index=False)
    return out
