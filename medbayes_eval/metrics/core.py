"""Core metrics suite for clinical uncertainty governance.

Operates on the standard per-sample record DataFrame (see schema.RECORD_COLUMNS).
All functions are pure and vectorized. The headline aggregator is
`compute_record_metrics(df, ...) -> dict` (flat scalars), and `metrics_table`
produces a tidy per-group table consumed by plotting/reporting.

Metric families:
  - calibration:   ECE, NLL, Brier
  - discrimination: AUROC/AUPRC (binary task) and AUROC of error detection
  - selective:     coverage@tau, selective/accepted accuracy, AURC, E-AURC
  - clinical risk: overconfident-error, harmful-overconfident-error,
                   severity-weighted error, abstention precision/recall
  - secondary:     CUS (severity-weighted calibration), ZTI (coverage/acc F-score)
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score

from .. import config
from ..schema import probs_matrix

_EPS = 1e-12


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _arrays(df: pd.DataFrame):
    """Extract the common arrays used by most metrics."""
    conf = df["confidence"].to_numpy(dtype=float)
    correct = df["correct"].to_numpy(dtype=bool)
    sev = df["severity"].to_numpy(dtype=float)
    labels = df["label"].to_numpy(dtype=int)
    return conf, correct, sev, labels


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------
def expected_calibration_error(conf: np.ndarray, correct: np.ndarray,
                               n_bins: int = config.ECE_BINS) -> float:
    """Standard ECE with equal-width confidence bins."""
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, float)
    if len(conf) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            ece += (mask.sum() / n) * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def negative_log_likelihood(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multiclass NLL using the calibrated predictive distribution."""
    probs = np.asarray(probs, float)
    labels = np.asarray(labels, int)
    if len(labels) == 0:
        return float("nan")
    p_true = probs[np.arange(len(labels)), labels]
    p_true = np.clip(p_true, _EPS, 1.0)
    return float(-np.mean(np.log(p_true)))


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multiclass Brier score: mean sum_k (p_k - 1[k=y])^2."""
    probs = np.asarray(probs, float)
    labels = np.asarray(labels, int)
    if len(labels) == 0:
        return float("nan")
    n, k = probs.shape
    onehot = np.zeros((n, k))
    onehot[np.arange(n), labels] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


# ---------------------------------------------------------------------------
# discrimination
# ---------------------------------------------------------------------------
def auroc_error_detection(conf: np.ndarray, correct: np.ndarray) -> float:
    """AUROC of using (1 - confidence) to detect incorrect predictions.

    Always applicable (task-agnostic). High = uncertainty separates errors well.
    """
    correct = np.asarray(correct, bool)
    if correct.all() or (~correct).all():
        return float("nan")
    score = 1.0 - np.asarray(conf, float)  # higher score => predicted "incorrect"
    return float(roc_auc_score((~correct).astype(int), score))


def auroc_auprc_label(probs: np.ndarray, labels: np.ndarray, n_classes: int):
    """Binary-task AUROC/AUPRC using P(positive). NaN for non-binary tasks."""
    if n_classes != 2:
        return float("nan"), float("nan")
    labels = np.asarray(labels, int)
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    p_pos = np.asarray(probs, float)[:, 1]
    return (float(roc_auc_score(labels, p_pos)),
            float(average_precision_score(labels, p_pos)))


# ---------------------------------------------------------------------------
# selective prediction / risk-coverage
# ---------------------------------------------------------------------------
def risk_coverage_curve(conf: np.ndarray, correct: np.ndarray):
    """Return (coverage, risk) arrays by accepting in descending-confidence order."""
    conf = np.asarray(conf, float)
    err = (~np.asarray(correct, bool)).astype(float)
    order = np.argsort(-conf, kind="mergesort")
    err_sorted = err[order]
    n = len(err_sorted)
    coverage = np.arange(1, n + 1) / n
    risk = np.cumsum(err_sorted) / np.arange(1, n + 1)
    return coverage, risk


def aurc(conf: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    if len(conf) == 0:
        return float("nan")
    coverage, risk = risk_coverage_curve(conf, correct)
    return float(np.trapezoid(risk, coverage) / (coverage[-1] - coverage[0] + _EPS)) \
        if len(coverage) > 1 else float(risk[0])


def excess_aurc(conf: np.ndarray, correct: np.ndarray) -> float:
    """E-AURC = AURC - AURC of the optimal (oracle) confidence ranking."""
    if len(conf) == 0:
        return float("nan")
    actual = aurc(conf, correct)
    # oracle: rank all correct first -> use `correct` itself as confidence proxy
    oracle = aurc(np.asarray(correct, float), correct)
    return float(actual - oracle)


def selective_at_threshold(conf, correct, sev, tau: float):
    """Coverage, accepted accuracy, accepted error, abstention rate at threshold."""
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, bool)
    accepted = conf >= tau
    n = len(conf)
    coverage = float(accepted.mean()) if n else float("nan")
    if accepted.any():
        acc_acc = float(correct[accepted].mean())
    else:
        acc_acc = float("nan")
    acc_err = (1.0 - acc_acc) if not np.isnan(acc_acc) else float("nan")
    abst_rate = float((~accepted).mean()) if n else float("nan")
    return coverage, acc_acc, acc_err, abst_rate


# ---------------------------------------------------------------------------
# clinical risk
# ---------------------------------------------------------------------------
def overconfident_error_rate(conf, correct, tau: float) -> float:
    """Fraction of ALL samples that are wrong AND confidently accepted (conf>=tau)."""
    conf = np.asarray(conf, float)
    wrong = ~np.asarray(correct, bool)
    if len(conf) == 0:
        return float("nan")
    return float(np.mean(wrong & (conf >= tau)))


def harmful_overconfident_error_rate(conf, correct, sev, tau: float,
                                     cutoff: float = config.HIGH_SEVERITY_CUTOFF) -> float:
    """Wrong AND confident AND high clinical severity (sev >= cutoff)."""
    conf = np.asarray(conf, float)
    wrong = ~np.asarray(correct, bool)
    sev = np.asarray(sev, float)
    if len(conf) == 0:
        return float("nan")
    return float(np.mean(wrong & (conf >= tau) & (sev >= cutoff)))


def severity_weighted_error(conf, correct, sev, tau: float) -> float:
    """Mean of severity * 1[wrong & accepted]; penalizes high-harm overconfidence."""
    conf = np.asarray(conf, float)
    wrong = (~np.asarray(correct, bool)).astype(float)
    sev = np.asarray(sev, float)
    if len(conf) == 0:
        return float("nan")
    return float(np.mean(sev * wrong * (conf >= tau)))


def abstention_precision_recall(conf, correct, sev, tau: float,
                                harm_target: bool = False,
                                cutoff: float = config.HIGH_SEVERITY_CUTOFF):
    """Quality of abstention decisions at threshold tau.

    target = samples that SHOULD be abstained:
      - error mode  (harm_target=False): wrong predictions
      - harm mode   (harm_target=True):  wrong AND high-severity

    precision = P(target | abstained), recall = P(abstained | target).
    """
    conf = np.asarray(conf, float)
    wrong = ~np.asarray(correct, bool)
    sev = np.asarray(sev, float)
    abstained = conf < tau
    if harm_target:
        target = wrong & (sev >= cutoff)
    else:
        target = wrong
    n_abst = abstained.sum()
    n_target = target.sum()
    precision = float((abstained & target).sum() / n_abst) if n_abst > 0 else float("nan")
    recall = float((abstained & target).sum() / n_target) if n_target > 0 else float("nan")
    return precision, recall


# ---------------------------------------------------------------------------
# secondary clinical summaries (paper definitions, operationalized)
# ---------------------------------------------------------------------------
def clinical_uncertainty_score(conf, correct, sev,
                               n_bins: int = config.ECE_BINS) -> float:
    """CUS: severity-weighted calibration error.

    Operationalizes the paper's CUS = sum_b |acc(b)-conf(b)| * P(b) * w_clinical(b)
    by binning on confidence and weighting each bin by the mean clinical severity
    of its samples (normalized so a severity-1 cohort reduces to ECE). Lower is
    better: it penalizes miscalibration most where clinical harm is highest.
    """
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, float)
    sev = np.asarray(sev, float)
    n = len(conf)
    if n == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    cus = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            gap = abs(correct[mask].mean() - conf[mask].mean())
            p_b = mask.sum() / n
            w_b = sev[mask].mean()  # severity-1 -> reduces to ECE
            cus += gap * p_b * w_b
    return float(cus)


def zero_shot_trust_index(conf, correct, tau: float) -> float:
    """ZTI = harmonic mean (F-score) of coverage and accepted accuracy at tau.

    Higher is better: rewards models that stay applicable (coverage) while being
    reliable when they do answer (accepted accuracy).
    """
    coverage, acc_acc, _, _ = selective_at_threshold(conf, correct, None, tau)
    if np.isnan(acc_acc) or (coverage + acc_acc) <= 0:
        return float("nan")
    return float(2.0 * coverage * acc_acc / (coverage + acc_acc))


# ---------------------------------------------------------------------------
# headline aggregator
# ---------------------------------------------------------------------------
def compute_record_metrics(
    df: pd.DataFrame,
    *,
    coverage_thresholds: Sequence[float] = config.COVERAGE_THRESHOLDS,
    overconf_thresholds: Sequence[float] = config.OVERCONF_THRESHOLDS,
    high_severity_cutoff: float = config.HIGH_SEVERITY_CUTOFF,
    zti_tau: float = 0.7,
) -> dict:
    """Compute the full flat metric dict for one homogeneous slice of records."""
    out: dict = {"n": int(len(df))}
    if len(df) == 0:
        return out
    conf, correct, sev, labels = _arrays(df)
    probs = probs_matrix(df)
    # impute any NaN padding in ragged probs with 0 for scoring
    probs = np.nan_to_num(probs, nan=0.0)
    n_classes = int(df["n_classes"].iloc[0])

    out["accuracy"] = float(correct.mean())
    out["ece"] = expected_calibration_error(conf, correct)
    out["nll"] = negative_log_likelihood(probs, labels)
    out["brier"] = brier_score(probs, labels)
    out["auroc_error"] = auroc_error_detection(conf, correct)
    al, ap = auroc_auprc_label(probs, labels, n_classes)
    out["auroc_label"], out["auprc_label"] = al, ap
    out["aurc"] = aurc(conf, correct)
    out["eaurc"] = excess_aurc(conf, correct)
    out["cus"] = clinical_uncertainty_score(conf, correct, sev)
    # ZTI per threshold; scalar `zti` falls back to the lowest threshold that yields
    # non-empty coverage so it stays defined for under-confident multi-class models.
    zti_vals = {}
    for tau in coverage_thresholds:
        zti_vals[tau] = zero_shot_trust_index(conf, correct, tau)
        out[f"zti@{tau:g}"] = zti_vals[tau]
    primary = zti_vals.get(zti_tau, float("nan"))
    if np.isnan(primary):
        for tau in sorted(coverage_thresholds):
            if not np.isnan(zti_vals[tau]):
                primary = zti_vals[tau]
                break
    out["zti"] = primary
    out["mean_confidence"] = float(conf.mean())
    out["mean_uncertainty"] = float(df["uncertainty"].mean())
    out["mean_severity"] = float(sev.mean())

    for tau in coverage_thresholds:
        cov, acc_acc, acc_err, abst = selective_at_threshold(conf, correct, sev, tau)
        prec, rec = abstention_precision_recall(conf, correct, sev, tau)
        hprec, hrec = abstention_precision_recall(conf, correct, sev, tau, harm_target=True)
        t = f"{tau:g}"
        out[f"cov@{t}"] = cov
        out[f"selacc@{t}"] = acc_acc
        out[f"accerr@{t}"] = acc_err
        out[f"abst_rate@{t}"] = abst
        out[f"abst_prec@{t}"] = prec
        out[f"abst_rec@{t}"] = rec
        out[f"abst_prec_harm@{t}"] = hprec
        out[f"abst_rec_harm@{t}"] = hrec

    for tau in overconf_thresholds:
        t = f"{tau:g}"
        out[f"overconf@{t}"] = overconfident_error_rate(conf, correct, tau)
        out[f"harmful@{t}"] = harmful_overconfident_error_rate(
            conf, correct, sev, tau, high_severity_cutoff)
        out[f"sevwerr@{t}"] = severity_weighted_error(conf, correct, sev, tau)
    return out


def metrics_table(
    df: pd.DataFrame,
    group_cols: Sequence[str] = ("dataset", "model", "method"),
    **kw,
) -> pd.DataFrame:
    """Group the records and compute the flat metric dict per group -> tidy table."""
    group_cols = [c for c in group_cols if c in df.columns]
    rows = []
    if not group_cols:
        m = compute_record_metrics(df, **kw)
        rows.append(m)
        return pd.DataFrame(rows)
    for keys, sub in df.groupby(group_cols, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rec = dict(zip(group_cols, keys))
        rec.update(compute_record_metrics(sub, **kw))
        rows.append(rec)
    return pd.DataFrame(rows)


def bucket_metrics_table(
    df: pd.DataFrame,
    group_cols: Sequence[str] = ("dataset", "model", "method"),
    **kw,
) -> pd.DataFrame:
    """Explode the JSON `buckets` column and compute metrics per bucket per group."""
    import json
    exploded = df.copy()
    exploded["bucket"] = exploded["buckets"].apply(json.loads)
    exploded = exploded.explode("bucket")
    cols = list(group_cols) + ["bucket"]
    return metrics_table(exploded, group_cols=cols, **kw)


# ---------------------------------------------------------------------------
# CUS / ZTI validation: correlate against clinically meaningful failure metrics
# ---------------------------------------------------------------------------
def correlation_analysis(
    metrics_df: pd.DataFrame,
    secondary: Sequence[str] = ("cus", "zti"),
    cus_targets: Sequence[str] = ("harmful@0.8", "sevwerr@0.8", "aurc", "abst_rec@0.7"),
    zti_targets: Sequence[str] = ("selacc@0.7", "cov@0.7", "aurc", "abst_prec@0.7"),
) -> pd.DataFrame:
    """Pearson + Spearman correlation of CUS/ZTI vs clinical metrics across groups.

    This is the evidence for whether CUS/ZTI deserve to be reported at all: if they
    do not track harmful overconfidence / selective reliability, they get demoted.
    """
    from scipy.stats import pearsonr, spearmanr
    rows = []
    target_map = {"cus": cus_targets, "zti": zti_targets}
    for sec in secondary:
        if sec not in metrics_df.columns:
            continue
        for tgt in target_map.get(sec, []):
            if tgt not in metrics_df.columns:
                continue
            sub = metrics_df[[sec, tgt]].dropna()
            if len(sub) < 3 or sub[sec].nunique() < 2 or sub[tgt].nunique() < 2:
                pr = sr = pp = sp = float("nan")
            else:
                pr, pp = pearsonr(sub[sec], sub[tgt])
                sr, sp = spearmanr(sub[sec], sub[tgt])
            rows.append({
                "secondary": sec, "target": tgt, "n": len(sub),
                "pearson_r": pr, "pearson_p": pp,
                "spearman_r": sr, "spearman_p": sp,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # known-answer sanity checks on synthetic data
    import json
    rng = np.random.default_rng(0)
    n = 2000
    k = 4
    labels = rng.integers(0, k, n)
    # well-calibrated-ish: correctness probability == confidence
    conf = rng.uniform(0.25, 0.99, n)
    correct = rng.uniform(size=n) < conf
    sev = rng.choice([1.0, 2.0, 3.0], size=n)
    probs = []
    preds = []
    for i in range(n):
        p = rng.uniform(0, 1, k)
        # make argmax match correctness intent loosely
        p = p / p.sum()
        probs.append(json.dumps([round(float(x), 6) for x in p]))
        preds.append(int(np.argmax(p)))
    df = pd.DataFrame({
        "dataset": "synth", "model": "m", "method": "base", "seed": 0,
        "uid": [str(i) for i in range(n)],
        "label": labels, "pred": preds, "n_classes": k,
        "confidence": conf, "uncertainty": 1 - conf, "probs": probs,
        "correct": correct, "buckets": json.dumps(["low_severity"]), "severity": sev,
    })
    m = compute_record_metrics(df)
    print("ECE (should be small, ~<0.05 for calibrated):", round(m["ece"], 4))
    print("accuracy:", round(m["accuracy"], 4), " AURC:", round(m["aurc"], 4),
          " E-AURC:", round(m["eaurc"], 4))
    print("harmful@0.8:", round(m["harmful@0.8"], 4),
          " overconf@0.8:", round(m["overconf@0.8"], 4))
    print("CUS:", round(m["cus"], 4), " ZTI:", round(m["zti"], 4))
    print("abst_prec@0.7:", round(m["abst_prec@0.7"], 4),
          " abst_rec@0.7:", round(m["abst_rec@0.7"], 4))
    # monotonicity check: AURC should be lower when confidence ranks errors well
    perfect_conf = correct.astype(float) * 0.5 + 0.5
    print("AURC(perfect ranking) <= AURC(random):",
          round(aurc(perfect_conf, correct), 4), "<=", round(m["aurc"], 4))
    tbl = metrics_table(df)
    print("metrics_table cols:", len(tbl.columns), "rows:", len(tbl))
