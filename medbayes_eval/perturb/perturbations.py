"""Controlled clinical perturbations for stress-testing LLM uncertainty.

Each perturbation takes a :class:`Sample` and returns a NEW perturbed
``Sample`` (deep-copied so the input is never mutated). The original ``uid``
and ``label`` are preserved so a perturbed item can be inner-joined back to its
clean counterpart; the kind of perturbation applied is recorded under
``meta['perturbation']`` and a boolean ``meta['perturb_applied']`` flags whether
the transform actually found something to change.

The perturbations are deliberately regex-based and side-effect free: if the
target pattern is absent from the text, the function returns an unchanged copy
with ``meta['perturb_applied'] = False`` rather than raising. This lets a caller
sweep every perturbation over a heterogeneous corpus without special-casing.

The module also provides :func:`perturbation_report`, which contrasts clean and
perturbed prediction-record frames and summarises how each perturbation shifts
confidence, entropy, abstention, prediction flips, and harmful over-confidence.
"""
from __future__ import annotations

import copy
import json
import re
from typing import Callable, Optional

import numpy as np
import pandas as pd

from medbayes_eval.schema import Sample


# Units that mark a number as a lab value or vital sign. Ordered roughly from
# most to least specific so the alternation matches the longest unit first.
_LAB_UNITS = [
    "mg/dl", "mg/dL", "g/dl", "g/dL", "mmol/l", "mmol/L", "mmol",
    "mEq/L", "meq/l", "mmHg", "mmhg", "bpm", "beats/min", "breaths/min",
    "U/L", "u/l", "IU/L", "ng/ml", "ng/mL", "pg/ml", "pg/mL",
    "mcg/dl", "µg/dl", "cells/uL", "cells/µL", "x10^9/L",
    "kg", "cm", "°C", "°F", "%",
]
# A number (int or decimal, optional sign) optionally separated from a unit by a
# single space. Used to find lab/vital readings such as "38.5°C" or "120 mmHg".
_NUM = r"[-+]?\d+(?:\.\d+)?"
_UNIT_ALT = "|".join(re.escape(u) for u in _LAB_UNITS)
_LAB_VALUE_RE = re.compile(rf"{_NUM}\s?(?:{_UNIT_ALT})")
# Any standalone number (used by alter_numeric_value).
_BARE_NUM_RE = re.compile(_NUM)
# Sentence terminator: a ., ! or ? that is followed by whitespace/end and is
# NOT a decimal point between digits (so "38.5" stays one token). The trailing
# punctuation is kept with the preceding sentence.
_SENT_BOUNDARY_RE = re.compile(r"(?<!\d)([.!?])(?=\s|$)")


def _split_sentences(text: str) -> list[str]:
    """Split ``text`` into trimmed, non-empty sentences (punctuation kept).

    Decimal points (e.g. in ``38.5``) are not treated as sentence boundaries.
    """
    # Insert a sentinel after each genuine terminator, then split on it.
    marked = _SENT_BOUNDARY_RE.sub(lambda m: m.group(1) + "\x00", text)
    return [s.strip() for s in marked.split("\x00") if s.strip()]


def _clone(sample: Sample, name: str, applied: bool) -> Sample:
    """Deep-copy ``sample`` and stamp the perturbation metadata."""
    new = copy.deepcopy(sample)
    if new.meta is None:
        new.meta = {}
    new.meta["perturbation"] = name
    new.meta["perturb_applied"] = bool(applied)
    return new


def _primary_field(sample: Sample) -> str:
    """Name of the text field a perturbation should target.

    Prefer ``context`` when it carries the clinical narrative, otherwise fall
    back to ``question``.
    """
    if sample.context and sample.context.strip():
        return "context"
    return "question"


def _get_field(sample: Sample, field: str) -> str:
    return getattr(sample, field) or ""


# --------------------------------------------------------------------------- #
# Perturbations
# --------------------------------------------------------------------------- #
def remove_key_evidence(sample: Sample, rng: np.random.Generator) -> Sample:
    """Drop the longest sentence from the primary clinical text.

    The longest sentence is a cheap proxy for the most information-dense clause,
    so removing it tends to strip the key piece of evidence a decision rests on.
    """
    name = "remove_key_evidence"
    field = _primary_field(sample)
    text = _get_field(sample, field)
    sentences = _split_sentences(text)
    if len(sentences) < 2:
        return _clone(sample, name, applied=False)

    longest = max(sentences, key=len)
    kept = [s for s in sentences if s is not longest]
    new = _clone(sample, name, applied=True)
    setattr(new, field, " ".join(kept))
    return new


def add_negation(sample: Sample, rng: np.random.Generator) -> Sample:
    """Negate a key clinical finding in the primary text.

    Prefixes the first sentence mentioning a finding/symptom/report verb with a
    denial, e.g. ``"The patient denies ..."``, flipping the polarity of the
    evidence without otherwise disturbing the narrative.
    """
    name = "add_negation"
    field = _primary_field(sample)
    text = _get_field(sample, field)
    sentences = _split_sentences(text)
    if not sentences:
        return _clone(sample, name, applied=False)

    finding_re = re.compile(
        r"\b(reports?|presents?|complains?|shows?|exhibits?|has|have|with|"
        r"fever|pain|cough|nausea|dyspnea|edema|elevated|positive)\b",
        re.IGNORECASE,
    )
    target_idx = next(
        (i for i, s in enumerate(sentences) if finding_re.search(s)), None
    )
    if target_idx is None:
        return _clone(sample, name, applied=False)

    target = sentences[target_idx]
    # Lower-case the first character so the inserted prefix reads naturally.
    body = target[0].lower() + target[1:] if target else target
    sentences[target_idx] = f"The patient denies that {body}"
    new = _clone(sample, name, applied=True)
    setattr(new, field, " ".join(sentences))
    return new


def add_contradictory_symptom(sample: Sample, rng: np.random.Generator) -> Sample:
    """Append a sentence asserting a contradictory finding.

    The appended clause is intentionally self-conflicting so a well-calibrated
    model should grow less certain when reconciling it with the original text.
    """
    name = "add_contradictory_symptom"
    field = _primary_field(sample)
    text = _get_field(sample, field)
    if not text.strip():
        return _clone(sample, name, applied=False)

    distractors = [
        "However, the patient also reports the opposite finding on re-examination.",
        "Notably, a repeat assessment showed the contradictory symptom instead.",
        "Conversely, the patient simultaneously denies and confirms the same finding.",
    ]
    addition = distractors[int(rng.integers(len(distractors)))]
    new = _clone(sample, name, applied=True)
    setattr(new, field, f"{text.rstrip()} {addition}")
    return new


def remove_lab_value(sample: Sample, rng: np.random.Generator) -> Sample:
    """Strip numeric lab/vital readings, replacing each with ``[value removed]``.

    Removes the quantitative anchor while leaving the surrounding clinical prose
    intact, so the model must reason without the measured value.
    """
    name = "remove_lab_value"
    field = _primary_field(sample)
    text = _get_field(sample, field)
    if not _LAB_VALUE_RE.search(text):
        return _clone(sample, name, applied=False)

    new_text = _LAB_VALUE_RE.sub("[value removed]", text)
    new = _clone(sample, name, applied=True)
    setattr(new, field, new_text)
    return new


def alter_numeric_value(sample: Sample, rng: np.random.Generator) -> Sample:
    """Scale the first numeric value by a clinically implausible factor.

    The first standalone number is multiplied by a factor drawn from
    ``{0.3, 3.0}`` (rng-controlled), pushing a plausible reading to an
    implausible one (e.g. a tripled potassium) without otherwise editing text.
    """
    name = "alter_numeric_value"
    field = _primary_field(sample)
    text = _get_field(sample, field)
    match = _BARE_NUM_RE.search(text)
    if match is None:
        return _clone(sample, name, applied=False)

    factor = float(rng.choice([0.3, 3.0]))
    original = float(match.group())
    altered = original * factor
    # Render as int when the source looked like an integer, else trim trailing
    # zeros from the decimal so "38.5" -> "115.5" not "115.50000001".
    if "." in match.group():
        altered_str = f"{altered:.2f}".rstrip("0").rstrip(".")
    else:
        altered_str = str(int(round(altered)))

    new_text = text[: match.start()] + altered_str + text[match.end():]
    new = _clone(sample, name, applied=True)
    setattr(new, field, new_text)
    return new


def add_distractor(sample: Sample, rng: np.random.Generator) -> Sample:
    """Append an irrelevant but medical-sounding distractor sentence."""
    name = "add_distractor"
    field = _primary_field(sample)
    text = _get_field(sample, field)
    if not text.strip():
        return _clone(sample, name, applied=False)

    distractors = [
        "The patient's grandmother had an unrelated history of seasonal allergies.",
        "Routine dental cleaning was completed last year without complications.",
        "The hospital cafeteria recently updated its low-sodium menu options.",
        "An incidental note mentions the patient prefers morning appointments.",
    ]
    addition = distractors[int(rng.integers(len(distractors)))]
    new = _clone(sample, name, applied=True)
    setattr(new, field, f"{text.rstrip()} {addition}")
    return new


def vague_evidence(sample: Sample, rng: np.random.Generator) -> Sample:
    """Replace specific clinical terms with vague paraphrases.

    Concrete measurements and named abnormalities are swapped for fuzzy
    descriptions (``"38.5°C"`` -> ``"a fever"``), eroding the precision of the
    evidence while keeping the gist.
    """
    name = "vague_evidence"
    field = _primary_field(sample)
    text = _get_field(sample, field)

    # Ordered (pattern, replacement) rules. Temperature -> "a fever",
    # dose-like mg amounts -> "a dose", named abnormal labs -> "some abnormal
    # labs", and any remaining lab value -> "an abnormal value".
    rules: list[tuple[re.Pattern, str]] = [
        (re.compile(rf"{_NUM}\s?°[CF]"), "a fever"),
        (re.compile(rf"{_NUM}\s?(?:mg|mcg|g)\b"), "a dose"),
        (re.compile(
            r"\belevated\s+(?:WBC|white blood cell count|WBCs|CRP|lactate|"
            r"creatinine|troponin)\b",
            re.IGNORECASE,
        ), "some abnormal labs"),
        (_LAB_VALUE_RE, "an abnormal value"),
    ]

    new_text = text
    changed = False
    for pattern, replacement in rules:
        new_text, n = pattern.subn(replacement, new_text)
        changed = changed or n > 0

    if not changed:
        return _clone(sample, name, applied=False)
    new = _clone(sample, name, applied=True)
    setattr(new, field, new_text)
    return new


# Public registry: perturbation name -> callable(sample, rng) -> Sample.
PERTURBATIONS: dict[str, Callable[[Sample, np.random.Generator], Sample]] = {
    "remove_key_evidence": remove_key_evidence,
    "add_negation": add_negation,
    "add_contradictory_symptom": add_contradictory_symptom,
    "remove_lab_value": remove_lab_value,
    "alter_numeric_value": alter_numeric_value,
    "add_distractor": add_distractor,
    "vague_evidence": vague_evidence,
}


def apply_perturbation(
    sample: Sample, name: str, rng: Optional[np.random.Generator] = None
) -> Sample:
    """Apply the named perturbation to ``sample`` and return a new ``Sample``.

    Parameters
    ----------
    sample:
        The clean item. It is never mutated.
    name:
        Key into :data:`PERTURBATIONS`.
    rng:
        Optional ``numpy`` generator for the stochastic perturbations. A fresh
        default generator is created when omitted.
    """
    if name not in PERTURBATIONS:
        raise KeyError(
            f"unknown perturbation {name!r}; valid: {sorted(PERTURBATIONS)}"
        )
    if rng is None:
        rng = np.random.default_rng()
    return PERTURBATIONS[name](sample, rng)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _shannon_entropy(probs_json: str) -> float:
    """Shannon entropy (nats) of a JSON-encoded probability list."""
    p = np.asarray(json.loads(probs_json), dtype=float)
    p = p[p > 0.0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def _group_column(df: pd.DataFrame) -> Optional[str]:
    """Return the column that names the perturbation group, if any."""
    for col in ("perturbation", "method"):
        if col in df.columns:
            return col
    return None


def perturbation_report(
    records_clean: pd.DataFrame,
    records_perturbed: pd.DataFrame,
    tau: float = 0.7,
) -> pd.DataFrame:
    """Summarise how each perturbation shifts model behaviour.

    The two frames carry standard record columns (``uid``, ``confidence``,
    ``uncertainty``, ``pred``, ``correct``, ``severity``, ``probs`` as JSON, and
    ``n_classes``) and share ``uid`` values. They are inner-joined on ``uid``;
    rows are then grouped by the perturbed frame's group column (``perturbation``
    or ``method``), or treated as a single group when neither exists.

    For each group the following are reported (perturbed minus clean unless the
    metric name says otherwise):

    * ``confidence_drop`` -- mean(conf_clean - conf_perturbed)
    * ``entropy_increase`` -- mean(H(probs_perturbed) - H(probs_clean))
    * ``abstention_increase`` -- mean(conf_perturbed<tau) - mean(conf_clean<tau)
    * ``flip_rate`` -- mean(pred_clean != pred_perturbed)
    * ``harmful_overconf_reduction`` -- mean(wrong & conf>=tau & severity>=2.5)
      on the clean frame minus the same quantity on the perturbed frame
    * ``n`` -- number of joined rows in the group
    """
    clean = records_clean.copy()
    pert = records_perturbed.copy()

    # Precompute per-row entropy and the harmful-overconfidence indicator.
    for df in (clean, pert):
        df["_entropy"] = df["probs"].apply(_shannon_entropy)
        wrong = ~df["correct"].astype(bool)
        df["_harmful"] = (
            wrong
            & (df["confidence"] >= tau)
            & (df["severity"] >= 2.5)
        ).astype(float)
        df["_abstain"] = (df["confidence"] < tau).astype(float)

    group_col = _group_column(pert)

    keep_clean = ["uid", "confidence", "pred", "_entropy", "_harmful", "_abstain"]
    keep_pert = keep_clean + ([group_col] if group_col else [])
    merged = pert[keep_pert].merge(
        clean[keep_clean], on="uid", suffixes=("_pert", "_clean")
    )

    def _summarise(g: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "confidence_drop": float(
                    (g["confidence_clean"] - g["confidence_pert"]).mean()
                ),
                "entropy_increase": float(
                    (g["_entropy_pert"] - g["_entropy_clean"]).mean()
                ),
                "abstention_increase": float(
                    g["_abstain_pert"].mean() - g["_abstain_clean"].mean()
                ),
                "flip_rate": float((g["pred_clean"] != g["pred_pert"]).mean()),
                "harmful_overconf_reduction": float(
                    g["_harmful_clean"].mean() - g["_harmful_pert"].mean()
                ),
                "n": int(len(g)),
            }
        )

    if group_col:
        out = (
            merged.groupby(group_col, sort=False)
            .apply(_summarise, include_groups=False)
            .reset_index()
            .rename(columns={group_col: "perturbation"})
        )
    else:
        row = _summarise(merged)
        row["perturbation"] = "all"
        out = pd.DataFrame([row])
        out = out[["perturbation", *row.index.drop("perturbation", errors="ignore")]]

    cols = [
        "perturbation",
        "confidence_drop",
        "entropy_increase",
        "abstention_increase",
        "flip_rate",
        "harmful_overconf_reduction",
        "n",
    ]
    return out[cols].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _demo_samples() -> list[Sample]:
    """A few synthetic clinical Samples covering the perturbation targets."""
    return [
        Sample(
            uid="s1",
            dataset="demo",
            question="A 64-year-old man presents to the ED.",
            label=1,
            n_classes=3,
            context=(
                "The patient reports crushing chest pain radiating to the left "
                "arm. Vitals show a temperature of 38.5°C and blood pressure of "
                "150 mmHg. Troponin is elevated at 2.5 ng/mL. He has a history "
                "of hypertension."
            ),
            meta={},
        ),
        Sample(
            uid="s2",
            dataset="demo",
            question=(
                "A 7-year-old girl has a fever of 39.2°C and an elevated WBC "
                "count. She was given 150 mg of acetaminophen."
            ),
            label=0,
            n_classes=2,
            options=["viral", "bacterial"],
            meta={},
        ),
        Sample(
            uid="s3",
            dataset="demo",
            question="Which diagnosis fits best?",
            label=2,
            n_classes=4,
            context=(
                "Serum potassium is 6.8 mmol/L and the patient complains of "
                "palpitations. The ECG shows peaked T waves."
            ),
            meta={},
        ),
    ]


def _demo_records(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tiny synthetic clean/perturbed record frames sharing uids."""

    def _row(uid, conf, probs, pred, correct, sev, method):
        return {
            "uid": uid,
            "confidence": conf,
            "uncertainty": float(_shannon_entropy(json.dumps(probs))),
            "pred": pred,
            "correct": correct,
            "severity": sev,
            "probs": json.dumps(probs),
            "n_classes": len(probs),
        }

    clean = pd.DataFrame(
        [
            _row("s1", 0.92, [0.04, 0.92, 0.04], 1, True, 3.0, None),
            _row("s2", 0.88, [0.88, 0.12], 0, True, 1.0, None),
            _row("s3", 0.81, [0.05, 0.05, 0.09, 0.81], 3, False, 2.8, None),
        ]
    )

    # Perturbed frame: lower confidence / flatter probs, a flip, and a group col.
    pert = pd.DataFrame(
        [
            {**_row("s1", 0.55, [0.30, 0.55, 0.15], 1, True, 3.0, None),
             "perturbation": "remove_key_evidence"},
            {**_row("s2", 0.60, [0.40, 0.60], 1, False, 1.0, None),
             "perturbation": "add_negation"},
            {**_row("s3", 0.45, [0.25, 0.25, 0.45, 0.05], 2, True, 2.8, None),
             "perturbation": "alter_numeric_value"},
        ]
    )
    return clean, pert


def _main() -> None:
    rng = np.random.default_rng(0)
    samples = _demo_samples()

    print("=" * 72)
    print("PERTURBATION DEMO")
    print("=" * 72)
    for sample in samples:
        print(f"\n### Sample {sample.uid} (label={sample.label})")
        print(f"CLEAN: {sample.display_text()}")
        for name in PERTURBATIONS:
            out = apply_perturbation(sample, name, rng)
            flag = out.meta.get("perturb_applied")
            print(f"  [{name}] applied={flag}")
            print(f"      -> {out.display_text()}")

    print("\n" + "=" * 72)
    print("PERTURBATION REPORT")
    print("=" * 72)
    clean_df, pert_df = _demo_records(rng)
    report = perturbation_report(clean_df, pert_df, tau=0.7)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(report.to_string(index=False))


if __name__ == "__main__":
    _main()
