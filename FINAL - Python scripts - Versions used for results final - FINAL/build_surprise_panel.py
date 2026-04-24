"""
build_surprise_panel.py

Reads macro_surprises_final.csv, extracts (releasedate, surp) pairs for every
macro indicator, aligns all series to a common start date (the latest first
observation across all indicators), then merges into a single panel where:
  - rows  = unique release dates (chronological)
  - cols  = one surp_ column per indicator
  - gaps  = 0  (no release on that date for that indicator)
"""

import os
import pandas as pd
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR   = os.path.join(_SCRIPT_DIR, '..', 'FINAL - Datasets used for final analysis')

# ── 1. Load raw data ──────────────────────────────────────────────────────────
raw = pd.read_csv(
    os.path.join(_DATA_DIR, "macro_surprises_final_fedincl.csv"),
    header=0,
    dtype=str,          # keep everything as strings initially
)

# ── 2. Identify indicators from column names ──────────────────────────────────
# Columns follow the pattern:  releasedate_X  /  actual_X  /  forecast_X  /  surp_X
date_cols = [c for c in raw.columns if c.startswith("releasedate_")]
surp_cols = [c for c in raw.columns if c.startswith("surp_")]

indicators = [c.replace("releasedate_", "") for c in date_cols]
print(f"Indicators found ({len(indicators)}): {indicators}\n")

# ── 3. Build one tidy DataFrame per indicator ─────────────────────────────────
def parse_date(s):
    """Parse DD.MM.YYYY string; return NaT on failure."""
    try:
        return pd.to_datetime(s.strip(), format="%d.%m.%Y")
    except Exception:
        return pd.NaT

def parse_surp(s):
    """Parse numeric surprise; return NaN for blanks / Excel errors."""
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    if s in ("", "NA", "#VALUE!", "#N/A"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan

series_dict = {}   # indicator -> pd.Series(surp, index=date)

for ind in indicators:
    dcol = f"releasedate_{ind}"
    scol = f"surp_{ind}"

    df_ind = pd.DataFrame({
        "date": raw[dcol].apply(parse_date),
        "surp": raw[scol].apply(parse_surp),
    })

    # drop rows where date or surp is missing
    df_ind = df_ind.dropna(subset=["date", "surp"])

    # deduplicate dates (keep first occurrence if any duplicates)
    df_ind = df_ind.drop_duplicates(subset="date").set_index("date")["surp"]
    df_ind = df_ind.sort_index()

    series_dict[ind] = df_ind

# ── 4. Find the latest start date across all indicators ───────────────────────
first_dates = {ind: s.index.min() for ind, s in series_dict.items()}
print("First available date per indicator:")
for ind, d in sorted(first_dates.items(), key=lambda x: x[1]):
    print(f"  {ind:30s}  {d.date()}")

common_start = max(first_dates.values())
print(f"\nCommon start date (latest first obs): {common_start.date()}")

# ── 5. Crop each series to the common start date ──────────────────────────────
for ind in indicators:
    series_dict[ind] = series_dict[ind][series_dict[ind].index >= common_start]

# ── 6. Merge into a single panel ──────────────────────────────────────────────
panel = pd.DataFrame({
    f"surp_{ind}": series_dict[ind] for ind in indicators
})

# Union of all release dates; fill absent releases with 0
panel = panel.sort_index().fillna(0)
panel.index.name = "date"

print(f"\nPanel shape: {panel.shape}  (dates x indicators)")
print(panel.head(10).to_string())

# ── 7. Save ───────────────────────────────────────────────────────────────────
out_path = os.path.join(_DATA_DIR, "macro_surprise_panel_fedincl.csv")
panel.to_csv(out_path)
print(f"\nSaved to: {out_path}")
