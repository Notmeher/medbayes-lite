# CUS Incremental-Value Analysis

**Question.** Does the Clinical Uncertainty Score (CUS) explain variance in clinical failure
*beyond* standard expected calibration error (ECE), or is it merely a re-encoding of ECE?

**Method.** Nested OLS regressions across 686 model/method/clinical-risk-category
groups (from the bucket-level metric table):
`target ~ ECE` (base) versus `target ~ ECE + CUS` (full), for two targets: the harmful
overconfident error rate and the severity-weighted error. We report R^2, the increment
Delta-R^2, a 2000-sample bootstrap CI for Delta-R^2, and a partial-F test for adding CUS. CUS
is a severity-weighted calibration error and is therefore collinear with ECE, so the partial-F
test and the Delta-R^2 CI are the primary evidence rather than the (variance-inflated) CUS
coefficient.

## Result (pooled)

| Target | R^2(ECE) | R^2(ECE+CUS) | Delta-R^2 | 95% CI | partial-F p |
|---|---|---|---|---|---|
| harmful overconfident error | 0.601 | 0.793 | **0.192** | [0.136, 0.248] | 2.9e-99 |
| severity-weighted error | 0.734 | 0.818 | **0.084** | [0.057, 0.114] | 3.1e-58 |

Per-model results (in `cus_incremental_value.csv`) are positive and significant in every model,
with Delta-R^2 for harmful overconfidence ranging from 0.086
to 0.539.

## Verdict

- harmful overconfident error: **CUS adds explanatory power beyond ECE**.
- severity-weighted error: **CUS adds explanatory power beyond ECE**.

CUS explains an additional ~19 percentage points of variance in harmful
overconfidence beyond ECE (and ~8 points for severity-weighted error),
both with bootstrap CIs excluding zero and partial-F p far below 0.05. This means CUS is not
redundant with ECE: it captures the clinical-severity structure of miscalibration that ECE,
which weights all errors equally, does not. CUS should be reported as a severity-aware
calibration metric that adds clinically meaningful information, while remaining interpretable as
a severity-weighted extension of ECE (the two are correlated by construction).
