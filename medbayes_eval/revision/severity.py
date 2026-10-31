"""R3:3 / R1:3 - robustness of the clinical-severity layer.

  severity_sensitivity  harmful@0.8 reduction (base -> MedBayes-Lite) under alternative
                        cutoffs, multi-category aggregation rules, a strict word-boundary
                        ruleset, uniform weights, and random +/-25% weights
  denominators          harmful errors per total / per high-severity / per accepted item
  multi_category        bucket-count distribution, co-occurrence, harm by #categories
  trigger_audit         substring-vs-word-boundary rule disagreement (false positives)
"""
from __future__ import annotations

import json
import re
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd

from .. import config
from ..buckets import rules
from ..stats.bootstrap import bootstrap_p
from .analyses import GATES, save
from .fastmetrics import ci
from .items import load_items

SETTINGS_METHODS = ("base", "temp_scaling", "medbayes_lite")


def _bucket_lists(dataset, strict=False):
    df = load_items(dataset, None)
    df = df[df.role != "train"].drop_duplicates("uid").set_index("uid")
    col = "buckets_strict" if strict else "buckets"
    return df[col].map(json.loads)


def _agg(buckets, weights, how):
    w = [weights.get(b, config.DEFAULT_SEVERITY) for b in buckets] or [config.DEFAULT_SEVERITY]
    return {"max": max(w), "mean": float(np.mean(w)), "sum": float(np.sum(w))}[how]


def _schemes(n_random=20, seed=0):
    W = config.SEVERITY_WEIGHTS
    out = []
    for how in ("max", "mean", "sum"):
        for c in config.SEVERITY_CUTOFFS:
            out.append({"scheme": f"aggregation={how}", "cutoff": c, "weights": W,
                        "how": how, "strict": False})
    for c in config.SEVERITY_CUTOFFS:
        out.append({"scheme": "strict word-boundary rules", "cutoff": c, "weights": W,
                    "how": "max", "strict": True})
    out.append({"scheme": "uniform weights (= overconfident error)", "cutoff": 1.0,
                "weights": {b: 1.0 for b in W}, "how": "max", "strict": False})
    rng = np.random.default_rng(seed)
    for i in range(n_random):
        out.append({"scheme": f"random +/-25% weights #{i}", "cutoff": 2.5,
                    "weights": {b: w * rng.uniform(0.75, 1.25) for b, w in W.items()},
                    "how": "max", "strict": False})
    return out


def _flags(preds, sev, tau=0.8, cutoff=2.5):
    return np.array([(~p.correct) & (p.score >= tau) & (sev >= cutoff) for p in preds],
                    float)


def severity_sensitivity(store, B: int = 2000, n_random: int = 20) -> pd.DataFrame:
    rows = []
    schemes = _schemes(n_random)
    for ds, model in sorted({(d, m) for d, m, _ in store.keys() if m != "qwen2_5_7b"}):
        cache = {}
        for gate in GATES:
            base = store.preds(ds, model, "base", gate)
            mbl = store.preds(ds, model, "medbayes_lite", gate)
            ts = store.preds(ds, model, "temp_scaling", gate)
            if not base or not mbl:
                continue
            uids = store.get(ds, model, "base")[0].uids
            idx = np.random.default_rng(0).integers(0, len(uids), (B, len(uids)))
            for sc in schemes:
                key = (sc["strict"], sc["how"], tuple(sorted(sc["weights"].items())))
                if key not in cache:
                    bl = _bucket_lists(ds, sc["strict"]).loc[uids]
                    cache[key] = bl.map(lambda b: _agg(b, sc["weights"], sc["how"])).to_numpy()
                sev = cache[key]
                fb = _flags(base, sev, cutoff=sc["cutoff"]).mean(0)
                fm = _flags(mbl, sev, cutoff=sc["cutoff"]).mean(0)
                ft = _flags(ts, sev, cutoff=sc["cutoff"]).mean(0) if ts else None
                d = fb[idx].mean(1) - fm[idx].mean(1)
                lo, hi = ci(d)
                rows.append({"dataset": ds, "model": model, "gate": gate,
                             "scheme": sc["scheme"], "cutoff": sc["cutoff"],
                             "frac_items_high_severity": float((sev >= sc["cutoff"]).mean()),
                             "harmful_base": fb.mean(), "harmful_temp_scaling":
                                 ft.mean() if ft is not None else np.nan,
                             "harmful_medbayes_lite": fm.mean(),
                             "reduction": fb.mean() - fm.mean(), "reduction_ci_lo": lo,
                             "reduction_ci_hi": hi, "p_value": bootstrap_p(d, B)})
    df = pd.DataFrame(rows)
    save(df, "severity_sensitivity")
    rand = df[df.scheme.str.startswith("random")]
    summ = (rand.groupby(["dataset", "model", "gate"])
            .agg(reduction_mean=("reduction", "mean"), reduction_min=("reduction", "min"),
                 reduction_max=("reduction", "max"),
                 all_ci_exclude_zero=("reduction_ci_lo", lambda s: bool((s > 0).all())))
            .reset_index())
    save(summ, "severity_sensitivity_random_weights_summary")
    return df


def denominators(thr: pd.DataFrame) -> pd.DataFrame:
    sub = thr[np.isclose(thr.tau, 0.8) & thr.method.isin(SETTINGS_METHODS)]
    cols = ["dataset", "model", "method", "gate", "n_items", "n_high_severity_items",
            "n_accepted_mean", "harmful_count_mean", "harmful_per_total",
            "harmful_per_high_severity_item", "harmful_per_accepted"]
    return save(sub[cols], "harmful_denominators",
                caption="harmful@0.8 under three denominators (3-seed pooled).",
                tex_cols=cols)


def multi_category(store) -> pd.DataFrame:
    rows, co_rows = [], []
    for ds in ("medmcqa", "medqa", "pubmedqa"):
        try:
            it = load_items(ds, "test")
        except FileNotFoundError:
            continue
        bl = it.buckets.map(json.loads)
        hist = Counter(min(len(b), 5) for b in bl)
        for k, v in sorted(hist.items()):
            rows.append({"dataset": ds, "n_categories": "5+" if k == 5 else str(k),
                         "n_items": v, "frac": v / len(bl)})
        co = Counter()
        for b in bl:
            for a, c in combinations(sorted(b), 2):
                co[(a, c)] += 1
        for (a, c), v in co.most_common():
            co_rows.append({"dataset": ds, "bucket_a": a, "bucket_b": c, "n_items": v})
    save(pd.DataFrame(rows), "multi_category_distribution")
    save(pd.DataFrame(co_rows), "bucket_cooccurrence")

    harm = []
    for ds, model in sorted({(d, m) for d, m, _ in store.keys() if m != "qwen2_5_7b"}):
        uids = store.get(ds, model, "base")[0].uids
        nb = _bucket_lists(ds).loc[uids].map(len).to_numpy()
        grp = np.where(nb >= 4, "4+", nb.astype(str))
        for method in SETTINGS_METHODS:
            for gate in GATES:
                preds = store.preds(ds, model, method, gate)
                if not preds:
                    continue
                f = _flags(preds, preds[0].sev).mean(0)
                for g in sorted(set(grp)):
                    m = grp == g
                    harm.append({"dataset": ds, "model": model, "method": method,
                                 "gate": gate, "n_categories": g, "n_items": int(m.sum()),
                                 "harmful@0.8_within_group": float(f[m].mean())})
    return save(pd.DataFrame(harm), "harmful_by_n_categories")


def trigger_audit() -> pd.DataFrame:
    """Per keyword cue: items matched by the default substring rule vs the strict
    word-boundary rule (substring-only hits are false-positive candidates)."""
    lex = {"medication_drug_interaction": rules._MEDICATION_KEYWORDS,
           "diagnosis_differential": rules._DIAGNOSIS_KEYWORDS,
           "treatment_recommendation": rules._TREATMENT_KEYWORDS,
           "risk_prognosis": rules._RISK_KEYWORDS,
           "lab_biomarker_interpretation": rules._LAB_KEYWORDS,
           "rare_disease": rules._RARE_KEYWORDS,
           "ambiguous_symptoms": rules._AMBIGUOUS_KEYWORDS,
           "contradictory_evidence": rules._CONTRADICTORY_KEYWORDS,
           "missing_evidence": rules._MISSING_KEYWORDS,
           "high_severity": rules._HIGH_SEVERITY_KEYWORDS}
    rows, agree = [], []
    for ds in ("medmcqa", "medqa", "pubmedqa"):
        try:
            it = load_items(ds, "test")
        except FileNotFoundError:
            continue
        texts = (it.question + " " + it.context.fillna("") + " " +
                 it.options.fillna("").map(lambda s: " ".join(json.loads(s)) if s else "")
                 ).str.lower().tolist()
        for bucket, kws in lex.items():
            for k in kws:
                sub = re.compile(re.escape(k))
                strict = rules._alt_strict([k])
                n_sub = sum(1 for t in texts if sub.search(t))
                n_str = sum(1 for t in texts if strict.search(t))
                if n_sub:
                    examples = Counter()
                    for t in texts:
                        for m in re.finditer(rf"\w*{re.escape(k)}\w*", t):
                            if not strict.search(m.group(0)):
                                examples[m.group(0)] += 1
                    rows.append({"dataset": ds, "bucket": bucket, "cue": k,
                                 "items_substring_match": n_sub, "items_word_match": n_str,
                                 "substring_only_items": n_sub - n_str,
                                 "substring_only_frac": (n_sub - n_str) / n_sub,
                                 "top_substring_only_tokens": ", ".join(
                                     w for w, _ in examples.most_common(5))})
        d = it.buckets.map(json.loads)
        s = it.buckets_strict.map(json.loads)
        for b in config.BUCKETS:
            a1 = d.map(lambda x: b in x).to_numpy()
            a2 = s.map(lambda x: b in x).to_numpy()
            agree.append({"dataset": ds, "bucket": b, "n_default": int(a1.sum()),
                          "n_strict": int(a2.sum()), "n_both": int((a1 & a2).sum()),
                          "jaccard": float((a1 & a2).sum() / max((a1 | a2).sum(), 1))})
        agree.append({"dataset": ds, "bucket": "ALL (severity changes)",
                      "n_default": int((it.severity >= 2.5).sum()),
                      "n_strict": int((it.severity_strict >= 2.5).sum()),
                      "n_both": int(((it.severity >= 2.5) & (it.severity_strict >= 2.5)).sum()),
                      "jaccard": np.nan})
    save(pd.DataFrame(agree), "ruleset_agreement_default_vs_strict")
    df = pd.DataFrame(rows).sort_values(["dataset", "substring_only_items"], ascending=[True, False])
    return save(df, "trigger_audit_false_positive_candidates")
