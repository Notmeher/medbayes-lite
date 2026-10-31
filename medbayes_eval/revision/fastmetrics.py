"""Vectorized metrics + item-level bootstrap engine for the revision analyses.

Definitions match medbayes_eval.metrics.core exactly (15 equal-width ECE bins with
(lo, hi] edges, AURC normalized by the coverage span, harmful = fraction of ALL items
that are wrong AND accepted AND high severity), but operate on numpy arrays so that
10k-draw paired bootstraps are tractable.

Every prediction set carries two scores:
  p      calibrated max probability (used for ECE / CUS, which need a probability)
  score  the gate score used for acceptance (max prob, or C(p) = 1 - H(p)/log K)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Sequence

import numpy as np

from .. import config

_EPS = 1e-12
N_BINS = config.ECE_BINS


def entropy(probs):
    p = np.clip(probs, _EPS, 1.0)
    return -(p * np.log(p)).sum(-1)


def normalized_conf(probs):
    k = probs.shape[-1]
    return 1.0 - entropy(probs) / np.log(k)


@dataclass
class Pred:
    """One prediction set (a method x seed on a fixed, uid-aligned item list)."""
    probs: np.ndarray      # (N, K)
    labels: np.ndarray     # (N,)
    sev: np.ndarray        # (N,) max bucket severity
    p: np.ndarray          # (N,) calibrated max probability
    score: np.ndarray      # (N,) gate score

    @property
    def correct(self):
        return self.probs.argmax(-1) == self.labels

    def take(self, idx):
        return Pred(self.probs[idx], self.labels[idx], self.sev[idx], self.p[idx],
                    self.score[idx])


def make_pred(probs, labels, sev, gate="maxprob", conf_override=None) -> Pred:
    p = probs.max(-1) if conf_override is None else np.asarray(conf_override, float)
    score = p if gate == "maxprob" else normalized_conf(probs)
    return Pred(np.asarray(probs, float), np.asarray(labels), np.asarray(sev, float), p, score)


# ---------------------------------------------------------------------------
# scalar metrics on arrays
# ---------------------------------------------------------------------------
def _bin_index(p):
    return np.clip(np.ceil(p * N_BINS).astype(int) - 1, 0, N_BINS - 1)


def ece(p, correct, weights=None):
    """ECE; with `weights` (per-item severity) it is CUS (bin weight = mean severity)."""
    n = len(p)
    if n == 0:
        return np.nan
    b = _bin_index(p)
    cnt = np.bincount(b, minlength=N_BINS)
    s_conf = np.bincount(b, weights=p, minlength=N_BINS)
    s_acc = np.bincount(b, weights=correct.astype(float), minlength=N_BINS)
    ok = cnt > 0
    gap = np.abs(s_acc[ok] - s_conf[ok]) / cnt[ok]
    w = cnt[ok] / n
    if weights is not None:
        w = w * (np.bincount(b, weights=weights, minlength=N_BINS)[ok] / cnt[ok])
    return float((gap * w).sum())


def nll(probs, labels):
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(labels)), labels], _EPS, 1))))


def brier(probs, labels):
    oh = np.zeros_like(probs)
    oh[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(((probs - oh) ** 2).sum(-1)))


def risk_coverage(score, correct):
    order = np.argsort(-score, kind="mergesort")
    err = (~correct[order]).astype(float)
    k = np.arange(1, len(err) + 1)
    return k / len(err), np.cumsum(err) / k


_trapz = getattr(np, "trapezoid", None) or np.trapz


def aurc(score, correct):
    if len(score) < 2:
        return np.nan
    cov, risk = risk_coverage(score, correct)
    return float(_trapz(risk, cov) / (cov[-1] - cov[0] + _EPS))


def auroc_error(score, correct):
    """AUROC of (1 - score) for detecting errors (Mann-Whitney, ties averaged)."""
    wrong = ~correct
    n1, n0 = wrong.sum(), correct.sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    from scipy.stats import rankdata
    r = rankdata(-score)
    return float((r[wrong].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def metrics(pr: Pred, taus=(0.5, 0.6, 0.7, 0.8, 0.9), harm_taus=(0.7, 0.8, 0.9),
            cutoff=config.HIGH_SEVERITY_CUTOFF, full=True) -> Dict[str, float]:
    c = pr.correct
    n = len(c)
    hi = pr.sev >= cutoff
    out = {"n": n, "accuracy": float(c.mean()), "ece": ece(pr.p, c),
           "cus": ece(pr.p, c, pr.sev), "aurc": aurc(pr.score, c),
           "n_high_sev": int(hi.sum())}
    if full:
        out.update(nll=nll(pr.probs, pr.labels), brier=brier(pr.probs, pr.labels),
                   auroc_error=auroc_error(pr.score, c),
                   mean_score=float(pr.score.mean()))
    for t in taus:
        acc = pr.score >= t
        na = int(acc.sum())
        wrong = ~c
        out[f"cov@{t:g}"] = na / n
        out[f"n_acc@{t:g}"] = na
        out[f"selacc@{t:g}"] = float(c[acc].mean()) if na else np.nan
        out[f"abst_rec@{t:g}"] = (float((~acc & wrong).sum() / wrong.sum())
                                  if wrong.any() else np.nan)
        out[f"abst_prec@{t:g}"] = float((~acc & wrong).sum() / (n - na)) if n - na else np.nan
    for t in harm_taus:
        flag = (~c) & (pr.score >= t)
        out[f"overconf@{t:g}"] = float(flag.mean())
        out[f"harmful@{t:g}"] = float((flag & hi).mean())
        out[f"harmful_count@{t:g}"] = int((flag & hi).sum())
        out[f"harmful_per_highsev@{t:g}"] = float((flag & hi).sum() / max(hi.sum(), 1))
        out[f"sevwerr@{t:g}"] = float(np.mean(pr.sev * flag))
    return out


# ---------------------------------------------------------------------------
# matched coverage / matched risk
# ---------------------------------------------------------------------------
def at_coverage(pr: Pred, coverage: float, cutoff=config.HIGH_SEVERITY_CUTOFF):
    """Accept the top ceil(coverage*N) items by gate score (stable order)."""
    n = len(pr.labels)
    k = int(np.ceil(coverage * n - 1e-9))
    order = np.argsort(-pr.score, kind="mergesort")
    acc = np.zeros(n, bool)
    acc[order[:k]] = True
    c = pr.correct
    wrong_acc = acc & ~c
    harm = wrong_acc & (pr.sev >= cutoff)
    return {"n_accepted": k, "coverage": k / n,
            "errors_accepted": int(wrong_acc.sum()),
            "selective_risk": float(wrong_acc.sum() / k) if k else np.nan,
            "selective_acc": float(c[acc].mean()) if k else np.nan,
            "harmful_accepted": int(harm.sum()),
            "harmful_per_total": float(harm.sum() / n),
            "harmful_among_accepted": float(harm.sum() / k) if k else np.nan,
            "sevwerr_per_total": float((pr.sev * wrong_acc).sum() / n)}


def coverage_at_risk(pr: Pred, risk: float):
    """Largest coverage whose selective risk (top-k by score) is <= `risk`."""
    cov, r = risk_coverage(pr.score, pr.correct)
    ok = np.where(r <= risk + 1e-12)[0]
    return float(cov[ok.max()]) if ok.size else 0.0


# ---------------------------------------------------------------------------
# bootstrap engine
# ---------------------------------------------------------------------------
def boot_indices(n: int, B: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(B, n))


def seed_mean(fn: Callable[[Pred], float], preds: Sequence[Pred], idx=None) -> float:
    vals = [fn(p if idx is None else p.take(idx)) for p in preds]
    return float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else np.nan


def bootstrap(fn, preds: Sequence[Pred], idx_mat: np.ndarray):
    """Seed-averaged statistic per bootstrap draw (items resampled, pairing kept)."""
    return np.array([seed_mean(fn, preds, idx) for idx in idx_mat])


def paired_bootstrap(fn, preds_a, preds_b, idx_mat):
    """Draws of seed-mean(fn on B) - seed-mean(fn on A) on the same resampled items."""
    return np.array([seed_mean(fn, preds_b, i) - seed_mean(fn, preds_a, i) for i in idx_mat])


def ci(draws, alpha=0.05):
    d = np.asarray(draws, float)
    d = d[~np.isnan(d)]
    if d.size == 0:
        return np.nan, np.nan
    lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(mid - half), float(mid + half)
