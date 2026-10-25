"""Feature-group test. Run:  python research.py
For each candidate group, compares walk-forward out-of-sample AUC against the base model on three targets:
  heat  = rest-of-season PPG 20%+ above early PPG
  cool  = 20%+ below
  beat  = 20%+ above what early PPG and last year's PPG alone predict (the target that removes the 'slow start' free lunch)
A group is kept if it improves 'beat' AUC with a 90% bootstrap lower bound above zero and costs heat/cool no more than 0.003.
"""
import sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import engine

K = int(sys.argv[1]) if len(sys.argv) > 1 else None
ctx = engine.prepare(everything=True)
K = K or min(ctx["K"], 14)
T = engine.build_training(ctx, K)
print(f"training table at cutoff Week {K}: {len(T)} player-seasons, {T.season.min()}-{T.season.max()}")
rng = np.random.default_rng(0)


def cols_for(groups):
    return engine.FEATS_BASE + [f for g in groups for f in engine.GROUPS[g]]


def evaluate(groups):
    wf, folds = engine.walk_forward(T, cols_for(groups))
    return wf, folds


def boot(a, b, n=600):
    """Paired bootstrap of AUC(b) - AUC(a) on the same rows."""
    y, pa, pb = a["y"], a["p"], b["p"]
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        d.append(roc_auc_score(y[i], pb[i]) - roc_auc_score(y[i], pa[i]))
    return np.percentile(d, [5, 50, 95])


base, folds = evaluate([])
print(f"folds {folds[0]}-{folds[-1]}")
rows = [dict(config="base", **{f"auc_{t}": roc_auc_score(base[t]["y"], base[t]["p"]) for t in base})]
keep = []
for g in engine.GROUPS:
    wf, _ = evaluate([g])
    r = dict(config=f"+{g}")
    for t in wf:
        r[f"auc_{t}"] = roc_auc_score(wf[t]["y"], wf[t]["p"])
        lo, mid, hi = boot(base[t], wf[t])
        r[f"d_{t}"], r[f"lo_{t}"], r[f"hi_{t}"] = r[f"auc_{t}"] - rows[0][f"auc_{t}"], lo, hi
    ok = r["lo_beat"] > 0 and r["d_heat"] >= -0.003 and r["d_cool"] >= -0.003
    r["keep"] = ok
    if ok:
        keep.append(g)
    rows.append(r)
wf, _ = evaluate(keep)
r = dict(config="+" + "+".join(keep) if keep else "base (nothing kept)")
for t in wf:
    r[f"auc_{t}"] = roc_auc_score(wf[t]["y"], wf[t]["p"])
    r[f"d_{t}"] = r[f"auc_{t}"] - rows[0][f"auc_{t}"]
rows.append(r)
R = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(R.round(3).to_string(index=False))
print("\nKEEP:", keep)
R.to_csv("research_results.csv", index=False)
