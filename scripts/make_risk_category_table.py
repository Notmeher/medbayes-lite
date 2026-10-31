"""Stage 1.3: emit the Clinical Risk Category table (LaTeX) from config, so severity
weights match the implementation exactly."""
import os
from medbayes_eval import config

# (category key, display name, rule examples, clinical motivation)
ROWS = [
    ("medication_drug_interaction", "Medication / drug interaction",
     "drug-name suffixes (-pril, -olol, -statin, -mab); dose amounts (e.g. 500\\,mg); ``interaction'', ``contraindicated''",
     "Dosing and interaction errors translate directly into patient harm."),
    ("diagnosis_differential", "Diagnosis / differential",
     "``diagnosis'', ``differential'', ``most likely'', ``consistent with''",
     "Misdiagnosis drives downstream management errors."),
    ("treatment_recommendation", "Treatment recommendation",
     "``treat'', ``therapy'', ``management'', ``first-line'', ``next step''",
     "An incorrect treatment recommendation is directly actionable and harmful."),
    ("risk_prognosis", "Risk / prognosis",
     "``prognosis'', ``mortality'', ``survival'', ``recurrence''",
     "Miscommunicated risk distorts shared decision-making."),
    ("rare_disease", "Rare / uncommon condition",
     "``rare'', ``atypical'', ``syndrome'', ``congenital''",
     "Rare presentations are high-uncertainty and high-stakes."),
    ("lab_biomarker_interpretation", "Laboratory / biomarker",
     "``serum'', ``creatinine'', ``WBC'', ``mg/dl'', ``titer''",
     "Lab misinterpretation propagates to diagnosis and dosing."),
    ("numeric_vital_lab_heavy", "Numeric / vital-heavy",
     "$\\geq 4$ numeric or unit tokens (mmHg, bpm, mg/dL)",
     "Numeric-dense items are prone to value misreading."),
    ("ambiguous_symptoms", "Ambiguous symptoms",
     "``nonspecific'', ``vague'', ``could be'', ``atypical presentation''",
     "Ambiguity warrants more conservative deferral."),
    ("missing_evidence", "Missing / incomplete evidence",
     "``not available'', ``unknown'', ``insufficient'', ``no history''",
     "Incomplete evidence should lower confidence."),
    ("contradictory_evidence", "Contradictory evidence",
     "``however'', ``despite'', ``inconsistent'', ``conflicting''",
     "Conflicting findings require cautious aggregation."),
    ("negation_heavy", "Negation-heavy text",
     "$\\geq 3$ negation cues (no, not, denies, absent)",
     "Negation flips clinical meaning and is error-prone."),
    ("high_severity", "High-severity case",
     "``acute'', ``severe'', ``critical'', ``ICU'', ``sepsis'', ``STAT''",
     "Emergent cases carry the highest cost of confident error."),
    ("low_severity", "Low-severity (routine)",
     "fallback when no higher-risk category fires",
     "Routine factual QA; baseline severity."),
]

lines = []
for key, name, rules, motiv in ROWS:
    w = config.SEVERITY_WEIGHTS.get(key, config.DEFAULT_SEVERITY)
    lines.append(f"{name} & {rules} & {w:.1f} & {motiv} \\\\")
body = "\n".join(lines)

tex = r"""\begin{table}[h]
\centering
\caption{Rule-based clinical risk categories used as evaluation strata. Each item is assigned to
one or more categories by transparent keyword and regular-expression rules over its text; the
severity weight is used to define harmful overconfident errors (severity $\geq 2.5$) and to weight
the Clinical Uncertainty Score. These categories are heuristic evaluation strata, not
ground-truth clinical labels.}
\label{tab:risk_categories}
\renewcommand{\arraystretch}{1.2}
\scriptsize
\begin{tabular}{p{2.7cm}p{5.3cm}p{1.1cm}p{3.4cm}}
\hline
\textbf{Clinical risk category} & \textbf{Rule examples} & \textbf{Sev.\ wt.} & \textbf{Clinical motivation} \\
\hline
""" + body + r"""
\hline
\end{tabular}
\end{table}
"""
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "paper", "rewrite", "risk_category_table.tex")
open(out, "w").write(tex)
print("wrote", out)
print(f"{len(ROWS)} categories; weights from config.SEVERITY_WEIGHTS")
