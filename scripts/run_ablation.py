"""Stage 1.2: component ablation on MedMCQA / PubMedBERT (strong, overconfident base).
Isolates the real components of the governance layer: MC-dropout averaging, temperature
calibration, and the abstention policy. The uncertainty-weighted attention component is a
lambda=0 placeholder in the released code and is NOT isolated here; see
attention_component_not_isolated.md.
"""
import os
import pandas as pd

from medbayes_eval import config
from medbayes_eval.pipeline import run_encoder_dataset
from medbayes_eval.metrics.core import compute_record_metrics

MET = os.path.join(config.RESULTS_DIR, "metrics")

# methods map to ablation variants:
#   base          -> Base (single pass)
#   mc_only       -> MC dropout only
#   temp_scaling  -> Temperature scaling only
#   abstain_only  -> Abstention only (base predictor + threshold policy)
#   medbayes_lite -> Full = MC dropout + temperature (+ abstention)
methods = ("base", "mc_only", "temp_scaling", "abstain_only", "medbayes_lite")
df = run_encoder_dataset("medmcqa", "pubmedbert_medmcqa", seeds=(0,),
                         methods=methods, mc_samples=15, mc_p=0.2, device="cuda")

LABEL = {
    "base": "Base",
    "mc_only": "MC dropout only",
    "temp_scaling": "Temperature scaling only",
    "abstain_only": "Abstention only",
    "medbayes_lite": "Full (MC + temperature + abstention)",
}
order = ["base", "mc_only", "temp_scaling", "abstain_only", "medbayes_lite"]
rows = []
for m in order:
    sub = df[df.method == m]
    if sub.empty:
        continue
    me = compute_record_metrics(sub)
    rows.append({"variant": LABEL[m], "accuracy": me["accuracy"], "ece": me["ece"],
                 "harmful@0.8": me["harmful@0.8"], "selacc@0.7": me["selacc@0.7"],
                 "cov@0.7": me["cov@0.7"], "aurc": me["aurc"]})
tbl = pd.DataFrame(rows)
tbl.to_csv(os.path.join(MET, "ablation_components.csv"), index=False)

# LaTeX
disp = tbl.copy()
for c in ["accuracy", "ece", "harmful@0.8", "selacc@0.7", "cov@0.7", "aurc"]:
    disp[c] = disp[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "--")
body = "\n".join(
    f"{r['variant']} & {r['accuracy']} & {r['ece']} & {r['harmful@0.8']} & {r['selacc@0.7']} & {r['cov@0.7']} & {r['aurc']} \\\\"
    for _, r in disp.iterrows())
tex = r"""\begin{table}[h]
\centering
\caption{Component ablation on MedMCQA (PubMedBERT). Temperature calibration is the dominant
lever for calibration and harmful-error reduction; MC-dropout averaging adds a smaller
improvement; the abstention-only variant equals the base predictor because abstention is a
threshold policy applied to the (here uncalibrated) confidence. The full layer is MC-dropout
plus temperature plus abstention. AURC is essentially unchanged across variants.}
\label{tab:ablation_components}
\renewcommand{\arraystretch}{1.15}
\small
\begin{tabular}{lcccccc}
\hline
\textbf{Variant} & \textbf{Acc} & \textbf{ECE}$\downarrow$ & \textbf{harmful@0.8}$\downarrow$ & \textbf{SelAcc@0.7}$\uparrow$ & \textbf{Cov@0.7} & \textbf{AURC}$\downarrow$ \\
\hline
""" + body + r"""
\hline
\end{tabular}
\end{table}
"""
open(os.path.join(MET, "ablation_components.tex"), "w").write(tex)
print(tbl.to_string(index=False))
print("\nwrote ablation_components.{csv,tex}")
