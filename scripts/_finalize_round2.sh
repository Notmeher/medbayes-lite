#!/bin/bash
cd /home/elias/elias_projects/LLMUncertainty
# 1. wait for dropout sweep CSV (last GPU output)
until [ -f results/metrics/sens_dropout_rate.csv ]; do sleep 15; done
sleep 3
echo "=== dropout sweep result ==="
column -t -s, results/metrics/sens_dropout_rate.csv
# 2. regenerate paper (writes main.tex + main_original.tex, builds appendix + 4 supp tables)
echo "=== regenerating paper ==="
python paper/rewrite/assemble_rewrite.py 2>&1 | tail -1
# 3. integrity + verification
python - <<'PY'
import re
s=open("paper/main.tex").read()
labels=set(re.findall(r'\\label\{([^}]+)\}',s)); refs=set(re.findall(r'\\ref\{([^}]+)\}',s))
print("dangling refs:", refs-labels or "NONE")
b=re.sub(r'(?m)%.*$','',s); import collections
bc=collections.Counter(re.findall(r'\\begin\{([^}]+)\}',b)); ec=collections.Counter(re.findall(r'\\end\{([^}]+)\}',b))
print("env imbalance:", {k:(bc[k],ec[k]) for k in set(bc)|set(ec) if bc[k]!=ec[k]} or "NONE")
print("appendix present:", '\\appendix' in s, "| app:attention label:", 'label{app:attention}' in s)
print("4 supp tables:", sum(t in s for t in ['tab:sens_mc','tab:sens_dropout','tab:sens_threshold','tab:sens_cus_weights']),"/4")
print("Algorithm cleaned (no attention step '2) Uncertainty-Weighted Attention'):", '2) Uncertainty-Weighted Attention' not in s)
print("attention subsubsection gone from body:", s.count('subsubsection{Uncertainty-Adjusted Attention')==0)
print("'API-compatible setting (closed models)':", s.count('API-compatible setting (closed models)'))
print("abstract softened:", 'can be overconfident' in s)
print("CUS 'pre-specified' supplementary:", 'pre-specified, clinically motivated supplementary summary' in s)
print("main==main_original:", open('paper/main.tex').read()==open('paper/main_original.tex').read())
PY
echo "=== FINALIZE DONE ==="
