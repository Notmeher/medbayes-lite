"""Table emitters for the clinical-LLM evaluation pipeline.

Every public function consumes a tidy metrics DataFrame (or a derived frame) and
writes a CSV plus a booktabs LaTeX table to ``outdir``. The LaTeX is built so it
drops straight into a paper: ``\\toprule``/``\\midrule``/``\\bottomrule`` rules,
escaped headers, and a ``\\caption``/``\\label`` pair.

The single low-level primitive is :func:`emit_table`; the higher-level helpers
(:func:`main_results_table`, :func:`bucket_table`, ...) are thin column-selection
wrappers around it. All selection is done on the *intersection* of requested and
available columns, so partial frames never raise ``KeyError``.
"""
from __future__ import annotations

import os
import re
from typing import Optional, Sequence

import pandas as pd

# Key columns surfaced in the headline results / ablation tables. Kept here so the
# main and ablation tables stay consistent with each other.
_KEY_METRIC_COLS = [
    "accuracy", "ece", "nll", "brier", "aurc",
    "harmful@0.8", "cov@0.7", "selacc@0.7",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sanitize_name(name: str) -> str:
    """Make a string safe for use as a file-name stem.

    Replaces the characters that show up in metric names (``@``, ``.``, ``/``)
    with underscores so e.g. ``'harmful@0.8'`` becomes ``'harmful_0_8'``.
    """
    return re.sub(r"[@./]", "_", str(name))


def _escape_latex(text: str) -> str:
    """Escape the LaTeX-special characters that occur in our headers/cells.

    We only escape what actually appears in metric/column names (underscores,
    ampersands, percent, hash, dollar). A pre-formatted ``mean ± std`` cell keeps
    its unicode ``±`` which modern LaTeX engines (or ``inputenc``) handle.
    """
    out = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    # Handle backslash first to avoid double-escaping the replacements above.
    out = out.replace("\\", replacements["\\"])
    for ch, rep in replacements.items():
        if ch == "\\":
            continue
        out = out.replace(ch, rep)
    return out


def _select_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Return ``df`` restricted to ``columns`` that are actually present.

    Order follows ``columns``; missing names are silently dropped so callers can
    pass an idealized column list against a partially populated frame.
    """
    present = [c for c in columns if c in df.columns]
    return df.loc[:, present].copy()


def _format_value(value, float_fmt: str) -> str:
    """Format a single cell for LaTeX rendering."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float):
        try:
            return float_fmt % value
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return str(value)
    return str(value)


def _apply_mean_std(df: pd.DataFrame, float_fmt: str) -> pd.DataFrame:
    """Collapse ``<col>`` / ``<col>_std`` pairs into ``"mean ± std"`` strings.

    Returns a new frame in which each base column that has a matching ``_std``
    sibling is replaced by a formatted string column, and the ``_std`` sibling is
    dropped. Columns without a sibling are left untouched.
    """
    out = df.copy()
    std_suffix = "_std"
    base_cols = [c for c in out.columns if not str(c).endswith(std_suffix)]
    for base in base_cols:
        std_col = f"{base}{std_suffix}"
        if std_col not in out.columns:
            continue

        def _fmt_pair(row, _base=base, _std=std_col):
            mean, std = row[_base], row[_std]
            if pd.isna(mean):
                return ""
            mean_s = _format_value(mean, float_fmt)
            if pd.isna(std):
                return mean_s
            std_s = _format_value(std, float_fmt)
            return f"{mean_s} ± {std_s}"

        out[base] = out.apply(_fmt_pair, axis=1)
        out = out.drop(columns=[std_col])
    return out


def _to_booktabs_latex(
    df: pd.DataFrame,
    caption: str,
    label: str,
    float_fmt: str,
    include_index: bool,
) -> str:
    """Render ``df`` as a standalone booktabs LaTeX ``table`` environment.

    Built manually (rather than via ``Styler.to_latex``) to guarantee consistent
    escaping, rule placement, and caption/label across pandas versions.
    """
    index_names = list(df.index.names) if include_index else []
    index_names = ["" if n is None else n for n in index_names]
    header_cells = index_names + [str(c) for c in df.columns]
    n_cols = len(header_cells)
    col_spec = "l" * n_cols if n_cols else "l"

    escaped_header = [_escape_latex(h) for h in header_cells]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
    ]
    if caption:
        lines.append(r"\caption{%s}" % _escape_latex(caption))
    if label:
        lines.append(r"\label{%s}" % label)
    lines.append(r"\begin{tabular}{%s}" % col_spec)
    lines.append(r"\toprule")
    lines.append(" & ".join(escaped_header) + r" \\")
    lines.append(r"\midrule")

    for idx, row in df.iterrows():
        cells = []
        if include_index:
            idx_vals = idx if isinstance(idx, tuple) else (idx,)
            cells.extend(_escape_latex(_format_value(v, float_fmt)) for v in idx_vals)
        for col in df.columns:
            cells.append(_escape_latex(_format_value(row[col], float_fmt)))
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Core emitter
# ---------------------------------------------------------------------------
def emit_table(
    df: pd.DataFrame,
    name: str,
    outdir: str,
    caption: str = "",
    label: str = "",
    columns: Optional[Sequence[str]] = None,
    index: Optional[Sequence[str] | str] = None,
    float_fmt: str = "%.3f",
    mean_std: bool = False,
) -> dict:
    """Write ``df`` to both ``<outdir>/<name>.csv`` and ``<outdir>/<name>.tex``.

    Parameters
    ----------
    df:
        Source frame.
    name:
        File-name stem; sanitized via :func:`sanitize_name`.
    outdir:
        Output directory (created if needed).
    caption, label:
        LaTeX ``\\caption`` / ``\\label`` content. ``label`` is emitted verbatim
        (so ``tab:foo`` stays intact); ``caption`` is LaTeX-escaped.
    columns:
        If given, restrict to this column order (intersection with present cols).
    index:
        If given, ``set_index`` on these column(s) before rendering.
    float_fmt:
        ``printf``-style format for floats (e.g. ``'%.3f'``).
    mean_std:
        If ``True``, collapse each ``<col>``/``<col>_std`` pair into a single
        ``"mean ± std"`` string column before LaTeX rendering. The CSV always
        keeps the raw (un-collapsed) columns.

    Returns
    -------
    dict
        ``{'csv': <path>, 'tex': <path>}``.
    """
    os.makedirs(outdir, exist_ok=True)
    stem = sanitize_name(name)

    work = df.copy()
    if columns is not None:
        # Preserve any index columns the caller also wants to set.
        wanted = list(columns)
        if index is not None:
            idx_cols = [index] if isinstance(index, str) else list(index)
            for ic in idx_cols:
                if ic not in wanted:
                    wanted.append(ic)
        # When collapsing to "mean ± std", keep each selected column's _std sibling
        # so the pairing in _apply_mean_std can find it.
        if mean_std:
            for col in list(wanted):
                std_col = f"{col}_std"
                if std_col in work.columns and std_col not in wanted:
                    wanted.append(std_col)
        work = _select_columns(work, wanted)

    if index is not None:
        idx_cols = [index] if isinstance(index, str) else list(index)
        idx_cols = [c for c in idx_cols if c in work.columns]
        if idx_cols:
            work = work.set_index(idx_cols)

    csv_path = os.path.join(outdir, f"{stem}.csv")
    tex_path = os.path.join(outdir, f"{stem}.tex")

    # CSV keeps the raw frame (with separate _std columns and the index).
    work.to_csv(csv_path, index=work.index.name is not None or work.index.nlevels > 1)

    latex_frame = _apply_mean_std(work, float_fmt) if mean_std else work
    include_index = (
        latex_frame.index.name is not None
        or latex_frame.index.nlevels > 1
        or index is not None
    )
    tex = _to_booktabs_latex(
        latex_frame,
        caption=caption,
        label=label,
        float_fmt=float_fmt,
        include_index=include_index,
    )
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex)

    return {"csv": csv_path, "tex": tex_path}


# ---------------------------------------------------------------------------
# High-level table builders
# ---------------------------------------------------------------------------
def main_results_table(metrics_df: pd.DataFrame, outdir: str, **kw) -> dict:
    """Emit the headline results table (``main_results``).

    Selects ``[dataset, model, method]`` plus the key calibration / selective-risk
    metrics that are present, and writes them through :func:`emit_table`.
    """
    cols = ["dataset", "model", "method"] + _KEY_METRIC_COLS
    kw.setdefault("caption", "Main results: calibration and selective-prediction metrics.")
    kw.setdefault("label", "tab:main_results")
    return emit_table(metrics_df, "main_results", outdir, columns=cols, **kw)


def bucket_table(
    metrics_df: pd.DataFrame, outdir: str, metric: str = "harmful@0.8", **kw
) -> dict:
    """Emit a bucket-by-method pivot of ``metric`` (``bucket_<metric>``).

    Rows are clinical buckets, columns are methods, values are ``metric``. If the
    frame lacks the required columns, an empty pivot is emitted rather than raising.
    """
    needed = {"bucket", "method", metric}
    if not needed.issubset(metrics_df.columns):
        pivot = pd.DataFrame()
    else:
        pivot = metrics_df.pivot_table(
            index="bucket", columns="method", values=metric, aggfunc="mean"
        )
        pivot.columns = [str(c) for c in pivot.columns]
        pivot = pivot.reset_index()

    name = f"bucket_{sanitize_name(metric)}"
    kw.setdefault("caption", f"Per-bucket {metric} by method.")
    kw.setdefault("label", f"tab:{name}")
    kw.setdefault("index", "bucket" if "bucket" in pivot.columns else None)
    return emit_table(pivot, name, outdir, **kw)


def ablation_table(metrics_df: pd.DataFrame, outdir: str, **kw) -> dict:
    """Emit an ablation table keyed by ``method`` variants (``ablation``)."""
    cols = ["method"] + _KEY_METRIC_COLS
    kw.setdefault("caption", "Ablation across method variants.")
    kw.setdefault("label", "tab:ablation")
    return emit_table(metrics_df, "ablation", outdir, columns=cols, **kw)


def ood_table(ood_df: pd.DataFrame, outdir: str, **kw) -> dict:
    """Emit an out-of-distribution transfer table (``ood``).

    Generic: selects all present columns (e.g. ``model``, ``transfer``,
    ``id_metric``, ``ood_metric``) in their given order.
    """
    kw.setdefault("caption", "Out-of-distribution transfer (ID vs OOD).")
    kw.setdefault("label", "tab:ood")
    return emit_table(ood_df, "ood", outdir, columns=list(ood_df.columns), **kw)


def correlation_table(corr_df: pd.DataFrame, outdir: str, **kw) -> dict:
    """Emit a CUS/ZTI-vs-clinical-metric correlation table (``cus_zti_correlation``)."""
    kw.setdefault("caption", "Correlation of CUS / ZTI with clinical metrics.")
    kw.setdefault("label", "tab:cus_zti_correlation")
    return emit_table(corr_df, "cus_zti_correlation", outdir, **kw)


# ---------------------------------------------------------------------------
# Self-test / demo
# ---------------------------------------------------------------------------
def _synthetic_metrics_df() -> pd.DataFrame:
    """Build a small synthetic tidy metrics frame for the self-test."""
    import numpy as np

    rng = np.random.default_rng(0)
    datasets = ["pubmedqa", "medmcqa"]
    methods = ["base", "temp_scaling", "medbayes_lite"]
    buckets = ["high_severity", "rare_disease", "negation_heavy"]

    metric_cols = [
        "accuracy", "ece", "nll", "brier", "aurc", "eaurc", "cus", "zti",
        "harmful@0.8", "overconf@0.8", "cov@0.7", "selacc@0.7",
        "abst_prec@0.7", "abst_rec@0.7", "sevwerr@0.8", "auroc_error",
    ]

    rows = []
    for ds in datasets:
        for m in methods:
            for bucket in buckets:
                row = {
                    "dataset": ds,
                    "model": "biobert_medmcqa",
                    "method": m,
                    "bucket": bucket,
                }
                for col in metric_cols:
                    row[col] = float(rng.uniform(0, 1))
                # A few _std siblings to exercise mean_std rendering.
                row["accuracy_std"] = float(rng.uniform(0, 0.05))
                row["ece_std"] = float(rng.uniform(0, 0.02))
                row["harmful@0.8_std"] = float(rng.uniform(0, 0.03))
                rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    outdir = "/tmp/table_test"
    df = _synthetic_metrics_df()

    results = {
        "main": main_results_table(df, outdir),
        "bucket": bucket_table(df, outdir, metric="harmful@0.8"),
        "ablation": ablation_table(df, outdir),
        "mean_std": emit_table(
            df,
            "mean_std_demo",
            outdir,
            columns=["dataset", "method", "accuracy", "ece", "harmful@0.8"],
            index=["dataset", "method"],
            caption="Mean +/- std demo.",
            label="tab:mean_std_demo",
            mean_std=True,
        ),
    }

    for key, paths in results.items():
        tex_path = paths["tex"]
        csv_path = paths["csv"]
        print(f"\n===== {key}: {tex_path} =====")
        with open(tex_path, encoding="utf-8") as fh:
            print(fh.read())
        assert os.path.exists(csv_path), f"missing CSV: {csv_path}"
        print(f"[ok] CSV exists: {csv_path}")

    print("\nAll tables emitted successfully.")
