"""
prediction_eval.py
------------------
Evaluates y_pred_val.csv and y_pred_test.csv against actual returns in
merged_announcement_panel_fedincl_nextday.csv.

Outputs (in outcomes/):
  eval_pearson_val.csv   — Pearson correlation, validation period
  eval_pearson_test.csv  — Pearson correlation, test period
  eval_hitrate_val.csv   — Hit rate (%), validation period
  eval_hitrate_test.csv  — Hit rate (%), test period
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR   = os.path.join(_SCRIPT_DIR, '..', 'FINAL - Datasets used for final analysis')
_OUT_DIR    = os.path.join(_SCRIPT_DIR, '..', 'outcomes')

# ── Load predictions ──────────────────────────────────────────────────────────
pred_val  = pd.read_csv(os.path.join(_OUT_DIR, "y_pred_val.csv"),  index_col=0, parse_dates=True)
pred_test = pd.read_csv(os.path.join(_OUT_DIR, "y_pred_test.csv"), index_col=0, parse_dates=True)

# ── Load actuals ──────────────────────────────────────────────────────────────
actuals = pd.read_csv(os.path.join(_DATA_DIR, "merged_announcement_panel_fedincl_nextday.csv"),
                      index_col=0, parse_dates=True)

# ── Parse column structure: model × target ────────────────────────────────────
# Column naming: "{model}_{target}"  e.g. "AR(1)_ret_CADUSD_cum3"
# Split on the first occurrence of "_ret_"
def parse_col(col):
    parts = col.split("_ret_", 1)
    return parts[0], "ret_" + parts[1]

pred_cols = list(pred_val.columns)
models  = list(dict.fromkeys(parse_col(c)[0] for c in pred_cols if "_ret_" in c))
targets = list(dict.fromkeys(parse_col(c)[1] for c in pred_cols if "_ret_" in c))

# Order targets: group by asset, then spot → cum3 → cum5 → cum10 → cum15
BASE_ASSETS = ["CADUSD","EURUSD","GBPUSD","GC","JPYUSD",
               "XLE","XLF","XLI","XLK","XLV","ZN","ZT","GSPC","RUT"]
HORIZONS    = ["", "_cum3", "_cum5", "_cum10", "_cum15"]
targets_ordered = [f"ret_{a}{h}" for h in HORIZONS for a in BASE_ASSETS
                   if f"ret_{a}{h}" in targets]
# append any leftover targets not matched above
targets_ordered += [t for t in targets if t not in targets_ordered]

# ── Compute metrics ───────────────────────────────────────────────────────────
def compute_metrics(pred_df):
    """Return (pearson_df, hitrate_df) with models as rows, targets as cols."""
    pearson = pd.DataFrame(index=models, columns=targets_ordered, dtype=float)
    hitrate = pd.DataFrame(index=models, columns=targets_ordered, dtype=float)

    for model in models:
        for target in targets_ordered:
            col = f"{model}_{target}"
            if col not in pred_df.columns:
                continue
            y_pred = pred_df[col].dropna()
            if target not in actuals.columns or len(y_pred) == 0:
                continue
            y_true = actuals[target].reindex(y_pred.index).dropna()
            y_pred = y_pred.reindex(y_true.index).dropna()
            y_true = y_true.reindex(y_pred.index)

            if len(y_pred) < 5:
                continue

            # Pearson
            r, _ = pearsonr(y_true, y_pred)
            pearson.loc[model, target] = round(r, 4)

            # Hit rate: sign(y_pred) == sign(y_true) on all dates;
            # y_pred == 0 counts as a miss (neutral call = wrong direction)
            hits = (np.sign(y_pred) == np.sign(y_true)).mean() * 100
            hitrate.loc[model, target] = round(hits, 2)

    pearson.index.name = "Model"
    hitrate.index.name = "Model"
    return pearson, hitrate

pearson_val,  hitrate_val  = compute_metrics(pred_val)
pearson_test, hitrate_test = compute_metrics(pred_test)

# ── % positive days (y_true > 0) per target, per period ──────────────────────
def pct_positive(pred_df):
    """One-row DataFrame: % of dates where actual return > 0."""
    row = {}
    for target in targets_ordered:
        if target not in actuals.columns:
            continue
        y_true = actuals[target].reindex(pred_df.index).dropna()
        if len(y_true) == 0:
            continue
        row[target] = round((y_true > 0).mean() * 100, 2)
    out = pd.DataFrame(row, index=["% Positive days"])
    out.index.name = "Model"
    return out

pct_pos_val  = pct_positive(pred_val)
pct_pos_test = pct_positive(pred_test)

# ── Save ──────────────────────────────────────────────────────────────────────
pearson_val.to_csv(os.path.join(_OUT_DIR, "eval_pearson_val.csv"))
pearson_test.to_csv(os.path.join(_OUT_DIR, "eval_pearson_test.csv"))
hitrate_val.to_csv(os.path.join(_OUT_DIR, "eval_hitrate_val.csv"))
hitrate_test.to_csv(os.path.join(_OUT_DIR, "eval_hitrate_test.csv"))
pct_pos_val.to_csv(os.path.join(_OUT_DIR, "eval_pct_positive_val.csv"))
pct_pos_test.to_csv(os.path.join(_OUT_DIR, "eval_pct_positive_test.csv"))

# ── Pretty print ──────────────────────────────────────────────────────────────
WIDTH = 70

def print_block(title, df):
    print("\n" + "="*WIDTH)
    print(title)
    print("="*WIDTH)
    print(df.to_string(float_format="%.4f"))

print_block("PEARSON CORRELATION — Validation",          pearson_val)
print_block("PEARSON CORRELATION — Test",                pearson_test)
print_block("HIT RATE (%) — Validation",                 hitrate_val)
print_block("HIT RATE (%) — Test",                       hitrate_test)
print_block("% POSITIVE DAYS (actual > 0) — Validation", pct_pos_val)
print_block("% POSITIVE DAYS (actual > 0) — Test",       pct_pos_test)

print(f"\nSaved: {_OUT_DIR}/eval_pearson_val.csv")
print(f"Saved: {_OUT_DIR}/eval_pearson_test.csv")
print(f"Saved: {_OUT_DIR}/eval_hitrate_val.csv")
print(f"Saved: {_OUT_DIR}/eval_hitrate_test.csv")
print(f"Saved: {_OUT_DIR}/eval_pct_positive_val.csv")
print(f"Saved: {_OUT_DIR}/eval_pct_positive_test.csv")
