import os
import pandas as pd
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR   = os.path.join(_SCRIPT_DIR, '..', 'FINAL - Datasets used for final analysis')
_OUT_DIR    = os.path.join(_SCRIPT_DIR, '..', 'outcomes')
os.makedirs(_OUT_DIR, exist_ok=True)

START = "2005-02-02"
END   = "2025-12-31"

# ── 1. ASSETS ────────────────────────────────────────────────────────────────
panel = pd.read_csv(os.path.join(_DATA_DIR, "merged_daily_panel_fedincl_nextday.csv"), parse_dates=["date"])
panel = panel[(panel["date"] >= START) & (panel["date"] <= END)]

# Keep only the spot-return columns (no _lag, _cum suffix)
asset_cols = [c for c in panel.columns
              if c.startswith("ret_") and not any(s in c for s in ("_lag", "_cum"))]

asset_stats = (
    panel[asset_cols]
    .agg(["count", "mean", "std", "min", "max"])
    .T
    .rename(columns={"count": "N", "mean": "Mean", "std": "Std", "min": "Min", "max": "Max"})
    .rename_axis("Variable")
    .reset_index()
)
asset_stats["N"] = asset_stats["N"].astype(int)

asset_stats.to_csv(os.path.join(_OUT_DIR, "desc_stats_assets.csv"), index=False, float_format="%.6f")
print(f"Saved: {_OUT_DIR}/desc_stats_assets.csv")
print(asset_stats.to_string(index=False))


# ── 2. SURPRISES + VIX ───────────────────────────────────────────────────────
surprises_raw = pd.read_csv(os.path.join(_DATA_DIR, "macro_surprises_final_fedincl.csv"), encoding="utf-8-sig")

# Indicator names (all columns that start with "surp_" in the surprises file,
# excluding release-date and actual/forecast columns)
surp_cols = [c for c in surprises_raw.columns if c.startswith("surp_")]
date_cols  = [c.replace("surp_", "releasedate_") for c in surp_cols]
# fedratedec has no separate date col – it shares the same row structure
# Check which date cols actually exist
available_date_cols = [c for c in date_cols if c in surprises_raw.columns]

rows = []
for surp_col in surp_cols:
    date_col = surp_col.replace("surp_", "releasedate_")
    if date_col not in surprises_raw.columns:
        # fedrate: use releasedate_fedratedec if present, else skip date filter
        date_col = None

    if date_col:
        tmp = surprises_raw[[date_col, surp_col]].copy()
        tmp[date_col] = pd.to_datetime(tmp[date_col], dayfirst=True, errors="coerce")
        tmp = tmp.dropna(subset=[date_col])
        tmp = tmp[(tmp[date_col] >= START) & (tmp[date_col] <= END)]
        series = pd.to_numeric(tmp[surp_col], errors="coerce").dropna()
    else:
        series = pd.to_numeric(surprises_raw[surp_col], errors="coerce").dropna()

    rows.append({
        "Variable": surp_col,
        "N":   len(series),
        "Mean": series.mean(),
        "Std":  series.std(),
        "Min":  series.min(),
        "Max":  series.max(),
    })

# VIX from the daily panel (already date-filtered above)
vix = panel["VIXCLS"].dropna()
rows.append({
    "Variable": "VIXCLS",
    "N":   len(vix),
    "Mean": vix.mean(),
    "Std":  vix.std(),
    "Min":  vix.min(),
    "Max":  vix.max(),
})

surp_stats = pd.DataFrame(rows)
surp_stats["N"] = surp_stats["N"].astype(int)

surp_stats.to_csv(os.path.join(_OUT_DIR, "desc_stats_surprises_vix.csv"), index=False, float_format="%.6f")
print(f"\nSaved: {_OUT_DIR}/desc_stats_surprises_vix.csv")
print(surp_stats.to_string(index=False))
