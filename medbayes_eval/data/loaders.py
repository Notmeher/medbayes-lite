"""Dataset loaders. Each returns a list[Sample] with a unified schema so the rest
of the pipeline is dataset-agnostic.

Available: PubMedQA (3-way), MedMCQA (4-way), MedQA-USMLE (4-way),
MediQ (4-way, clinical atomic-facts reformatting of MedQA). MIMIC is a gated
adapter stub (credentialed access; not run here).
"""
from __future__ import annotations

from typing import List, Optional
import hashlib

from datasets import load_dataset

from ..schema import Sample
from .. import config

PUBMEDQA_LABELS = {"yes": 0, "no": 1, "maybe": 2}


def _load(hf_id: str, *args):
    return load_dataset(hf_id, *args, revision=config.dataset_revision(hf_id))


def _cap(samples: List[Sample], max_samples: Optional[int]) -> List[Sample]:
    return samples[:max_samples] if max_samples else samples


def _det_split(n: int, n_test: int, seed: int = 0):
    """Deterministic index split into (calibration_idx, test_idx)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    test_idx = set(perm[:n_test].tolist())
    return test_idx


def load_pubmedqa(split: str = "test", max_samples: Optional[int] = None,
                  n_test: int = 500, seed: int = 0) -> List[Sample]:
    """PubMedQA pqa_labeled. Only a 1000-item 'train' exists upstream; we make a
    deterministic test/calibration split. split='test' -> held-out test,
    split='calibration' -> the complement (used to fit temperature scaling)."""
    ds = _load("qiaojin/PubMedQA", "pqa_labeled")["train"]
    test_idx = _det_split(len(ds), n_test, seed)
    samples = []
    for i, ex in enumerate(ds):
        is_test = i in test_idx
        if (split == "test") != is_test:
            continue
        ctx = " ".join(ex["context"]["contexts"]) if ex.get("context") else ""
        label = PUBMEDQA_LABELS.get(str(ex["final_decision"]).strip().lower(), 2)
        samples.append(Sample(
            uid=f"pubmedqa-{ex['pubid']}", dataset="pubmedqa",
            question=ex["question"], context=ctx, options=None,
            label=label, n_classes=3,
            meta={"final_decision": ex["final_decision"]},
        ))
    return _cap(samples, max_samples)


def load_medmcqa(split: str = "validation", max_samples: Optional[int] = None) -> List[Sample]:
    """MedMCQA 4-way single-best-answer. Validation split is labeled (test is hidden)."""
    ds = _load("openlifescienceai/medmcqa")[split]
    samples = []
    for ex in ds:
        options = [ex["opa"], ex["opb"], ex["opc"], ex["opd"]]
        samples.append(Sample(
            uid=f"medmcqa-{ex['id']}", dataset="medmcqa",
            question=ex["question"], context=None, options=options,
            label=int(ex["cop"]), n_classes=4,
            meta={"subject": ex.get("subject_name"), "choice_type": ex.get("choice_type")},
        ))
    return _cap(samples, max_samples)


def _letter_to_idx(letter: str) -> int:
    return {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}.get(str(letter).strip().upper(), 0)


def load_medqa(split: str = "test", max_samples: Optional[int] = None) -> List[Sample]:
    """MedQA-USMLE 4-options."""
    ds = _load("GBaker/MedQA-USMLE-4-options")[split]
    samples = []
    for i, ex in enumerate(ds):
        opts_dict = ex["options"]
        keys = sorted(opts_dict.keys())
        options = [opts_dict[k] for k in keys]
        label = _letter_to_idx(ex["answer_idx"])
        uid = f"medqa-{hashlib.md5(ex['question'].encode()).hexdigest()[:12]}"
        samples.append(Sample(
            uid=uid, dataset="medqa", question=ex["question"], context=None,
            options=options, label=label, n_classes=4,
            meta={"meta_info": ex.get("meta_info")},
        ))
    return _cap(samples, max_samples)


def load_mediq(split: str = "test", max_samples: Optional[int] = None) -> List[Sample]:
    """MediQ: clinical atomic-facts reformatting of MedQA-USMLE (4-way). Carries the
    patient vignette in `context`, which is useful for clinical-note framing and
    evidence-removal perturbations."""
    ds = load_dataset("stellalisy/mediq")[split]
    samples = []
    for ex in ds:
        opts_dict = ex["options"]
        keys = sorted(opts_dict.keys())
        options = [opts_dict[k] for k in keys]
        ctx_list = ex.get("context") or []
        ctx = " ".join(ctx_list) if isinstance(ctx_list, list) else str(ctx_list)
        label = _letter_to_idx(ex["answer_idx"])
        samples.append(Sample(
            uid=f"mediq-{ex['id']}", dataset="mediq", question=ex["question"],
            context=ctx, options=options, label=label, n_classes=4,
            meta={"atomic_facts": ex.get("atomic_facts")},
        ))
    return _cap(samples, max_samples)


def load_mimic(split: str = "test", max_samples: Optional[int] = None,
               path: Optional[str] = None) -> List[Sample]:
    """MIMIC-III in-hospital mortality adapter (credentialed access).

    Not run in this environment (no PHI). Provide a prepared JSONL at `path` with
    fields {uid, text, label} to enable. Raises a clear error otherwise so the
    pipeline degrades gracefully rather than fabricating clinical results.
    """
    if path is None:
        raise FileNotFoundError(
            "MIMIC-III requires credentialed PhysioNet access. Provide a prepared "
            "JSONL via path=... with {uid,text,label(0/1)} to enable this adapter.")
    import json
    samples = []
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            samples.append(Sample(
                uid=str(ex["uid"]), dataset="mimic", question=ex["text"],
                context=None, options=None, label=int(ex["label"]), n_classes=2,
                meta={}))
    return _cap(samples, max_samples)


LOADERS = {
    "pubmedqa": load_pubmedqa,
    "medmcqa": load_medmcqa,
    "medqa": load_medqa,
    "mediq": load_mediq,
    "mimic": load_mimic,
}


def load_dataset_samples(name: str, **kw) -> List[Sample]:
    if name not in LOADERS:
        raise KeyError(f"unknown dataset '{name}'. available: {list(LOADERS)}")
    return LOADERS[name](**kw)


if __name__ == "__main__":
    for name in ["pubmedqa", "medmcqa", "medqa", "mediq"]:
        s = load_dataset_samples(name, max_samples=3)
        print(f"\n== {name}: loaded {len(s)} (showing first) ==")
        x = s[0]
        print("uid:", x.uid, "| label:", x.label, "| n_classes:", x.n_classes)
        print("question:", x.question[:90])
        if x.options:
            print("options:", [o[:30] for o in x.options])
        print("text len:", len(x.display_text()))
