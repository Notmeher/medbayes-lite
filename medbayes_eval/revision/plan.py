"""Job lists for the revision runs (one source of truth for notebook and CLI)."""
from __future__ import annotations

from typing import Optional, Sequence

MCQ_MODELS = ("pubmedbert_medmcqa", "biobert_medmcqa")
MCQ_DATASETS = ("medmcqa", "medqa")
PUBMEDQA_MODELS = ("bert_base", "biobert", "clinicalbert", "pubmedbert")


def encoder_jobs(seeds: Sequence[int] = (0, 1, 2), mc_samples: int = 15, mc_p: float = 0.2,
                 max_eval: Optional[int] = None, max_train: Optional[int] = None,
                 include_sensitivity: bool = True, include_perturb: bool = True,
                 perturb_n: int = 300) -> list:
    common = dict(mc_samples=mc_samples, mc_p=mc_p, max_eval=max_eval)
    jobs = []
    for model in MCQ_MODELS:
        for ds in MCQ_DATASETS:
            for s in seeds:
                jobs.append({"id": f"logits__{ds}__{model}__seed{s}", "kind": "encoder_logits",
                             "cost": 4 if ds == "medmcqa" else 1.5,
                             "args": dict(dataset=ds, model=model, seed=s, **common)})
    for model in PUBMEDQA_MODELS:
        for s in seeds:
            jobs.append({"id": f"logits__pubmedqa__{model}__seed{s}", "kind": "encoder_logits",
                         "cost": 2,
                         "args": dict(dataset="pubmedqa", model=model, seed=s,
                                      max_train=max_train, **common)})
    if include_sensitivity:
        jobs.append({"id": "sensitivity__medmcqa__pubmedbert_medmcqa", "kind": "sensitivity",
                     "cost": 8, "args": dict(max_eval=max_eval)})
    if include_perturb:
        for ds in MCQ_DATASETS:
            jobs.append({"id": f"perturb__{ds}__pubmedbert_medmcqa", "kind": "perturb",
                         "cost": 3, "args": dict(dataset=ds, n=perturb_n, seeds=tuple(seeds),
                                                 mc_samples=mc_samples, mc_p=mc_p)})
    return jobs


def latency_jobs(reps: int = 30) -> list:
    return [{"id": "latency_benchmark", "kind": "latency", "cost": 1,
             "args": dict(reps=reps)}]


def generative_jobs(n: int = 200, n_samples: int = 10, batch_size: int = 8) -> list:
    return [{"id": "generative__qwen2_5_7b", "kind": "generative", "cost": 1,
             "args": dict(n=n, n_samples=n_samples, batch_size=batch_size)}]
