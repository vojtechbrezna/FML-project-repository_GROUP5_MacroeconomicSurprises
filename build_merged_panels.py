"""
build_merged_panels.py

Builds two analysis-ready datasets:

Dataset 1 — Daily structure (merged_daily_panel_nextday.csv)
    date | ret_* | ret_*_lag1 (AR1) | ret_*_cum5/10/15 | VIXCLS | surp_*

Dataset 2 — Release-date structure (merged_announcement_panel_fedincl_nextday.csv)
    date | surp_* | ret_* | ret_*_lag1 | VIXCLS | ret_*_cum5/10/15

    ret_* in the announcement panel = log(Close_{t+1} / Close_t), i.e. the
    close-to-close return from announcement date t to the next trading day t+1.
    This is a tradable return: a trader observing the surprise at the close of t
    can enter at t's close and exit at t+1's close.

    ret_*_lag1 = log(Close_t / Close_{t-1}), the return on the announcement day
    itself, included as an AR(1) predictor.

Surprise panel starts 2001-06-13 (latest first obs of any surprise series).
CADUSD / EURUSD / GBPUSD first available 2003-12-01, so the merged datasets
are cropped to that common start; surprise data before that date is dropped.

Assets: CADUSD, EURUSD, GBPUSD, GC, JPYUSD, XLE, XLF, XLI, XLK, XLV, ZN, ZT, GSPC, RUT
Returns: log(Close_t / Close_{t-1})
Cumulative forward returns: sum of log-returns from t+1 through t+H  (H = 5, 10, 15)
AR(1): lag-1 of the daily return (both datasets)
"""

import pandas as pd
import numpy as np

# ── 0. Constants ──────────────────────────────────────────────────────────────
SURP_START   = pd.Timestamp("2001-06-13")   # surprise panel start
END          = pd.Timestamp("2025-12-31")

# Raw ticker → clean label (matching merged_panel_announcements convention)
TICKER_MAP = {
    "CADUSD=X": "CADUSD",
    "EURUSD=X": "EURUSD",
    "GBPUSD=X": "GBPUSD",
    "GC=F":     "GC",
    "JPYUSD=X": "JPYUSD",
    "XLE":      "XLE",
    "XLF":      "XLF",
    "XLI":      "XLI",
    "XLK":      "XLK",
    "XLV":      "XLV",
    "ZN=F":     "ZN",
    "ZT=F":     "ZT",
    "^GSPC":    "GSPC",
    "^RUT":     "RUT",
}

CUM_HORIZONS = [3, 5, 10, 15]

# ── 1. Load asset close prices ────────────────────────────────────────────────
print("Loading asset prices...")
raw = pd.read_csv("assets_yf_data.csv", header=[0, 1], index_col=0, skiprows=[2])
raw.index = pd.to_datetime(raw.index)
raw.index.name = "date"

close_raw = raw.xs("Close", axis=1, level=0).copy()
close_raw.columns = [TICKER_MAP.get(c, c) for c in close_raw.columns]
close_raw = close_raw.apply(pd.to_numeric, errors="coerce")

# Crop to window (keep full history before START for lag/cum computation)
close_raw = close_raw.loc[:END]

# Forward-fill missing prices: different markets (FX, futures, equities) have
# different holiday schedules.  A closed market = 0 return on that day.
close_raw = close_raw.ffill()

# Determine actual common start: latest first-valid date across all assets
first_valid = {c: close_raw[c].first_valid_index() for c in close_raw.columns}
asset_common_start = max(first_valid.values())
START = max(SURP_START, asset_common_start)
print(f"Asset first-valid dates:")
for c, d in sorted(first_valid.items(), key=lambda x: x[1]):
    print(f"  {c:8s}  {d.date()}")
print(f"Common start for merged datasets: {START.date()}")
print(f"End: {END.date()}")

# ── 2. Compute daily log returns ──────────────────────────────────────────────
ret = np.log(close_raw / close_raw.shift(1))
ret.columns = [f"ret_{c}" for c in close_raw.columns]

# Next-day return: log(Close_{t+1} / Close_t) — tradable on announcement day t
ret_nextday = ret.shift(-1)   # same column names, shifted forward by one day

# ── 3. Compute forward cumulative returns (t+1 through t+H) ──────────────────
# cum_H at date t = sum of ret[t+1 : t+H]  →  shift(-H) rolling(H).sum() aligned
# We compute as: for each row t, look forward H steps.
# Implementation: rolling sum of the *future* returns using shift.
# cum_H[t] = ret[t+1] + ret[t+2] + ... + ret[t+H]
#           = rolling(H).sum() shifted backwards by 1

cum_frames = {}
asset_cols = list(TICKER_MAP.values())
for H in CUM_HORIZONS:
    # rolling(H).sum() is sum over the past H obs → shift by -(H) to align to t, then shift(-1) for t+1
    rolling_sum = ret.rolling(H).sum().shift(-(H))   # sum of t+1..t+H aligned at t
    rolling_sum.columns = [c.replace("ret_", f"ret_") + f"_cum{H}" for c in rolling_sum.columns]
    cum_frames[H] = rolling_sum

# ── 4. AR(1): lag-1 of daily return ──────────────────────────────────────────
lag1 = ret.shift(1)
lag1.columns = [c + "_lag1" for c in ret.columns]

# ── 5. Load VIX ───────────────────────────────────────────────────────────────
print("Loading VIX...")
vix = pd.read_excel("VIX.xlsx", sheet_name="Daily, Close")
vix["observation_date"] = pd.to_datetime(vix["observation_date"])
vix = vix.set_index("observation_date")[["VIXCLS"]].apply(pd.to_numeric, errors="coerce")
vix.index.name = "date"

# ── 6. Load surprise panel ────────────────────────────────────────────────────
print("Loading surprise panel...")
surp = pd.read_csv("macro_surprise_panel_fedincl.csv", index_col=0, parse_dates=True)
surp.index.name = "date"

# ── 7. Assemble the daily base frame ─────────────────────────────────────────
# Crop returns to [START, END]
ret_cropped  = ret.loc[START:END]
lag1_cropped = lag1.loc[START:END]
cum_cropped  = {H: cum_frames[H].loc[START:END] for H in CUM_HORIZONS}
vix_cropped  = vix.loc[START:END]

# Align on the return index (trading days)
daily = ret_cropped.copy()
daily = daily.join(lag1_cropped, how="left")
for H in CUM_HORIZONS:
    daily = daily.join(cum_cropped[H], how="left")
daily = daily.join(vix_cropped, how="left")

# Merge surprises → fill zeros where no release occurred
surp_cropped = surp.loc[START:END]
daily = daily.join(surp_cropped, how="left")
surp_cols = surp.columns.tolist()
daily[surp_cols] = daily[surp_cols].fillna(0)

# ── 8. Column order for Dataset 1 ────────────────────────────────────────────
ret_cols  = [c for c in daily.columns if c.startswith("ret_") and "_lag1" not in c and "_cum" not in c]
lag_cols  = [c for c in daily.columns if c.endswith("_lag1")]
cum_cols  = [c for c in daily.columns if "_cum" in c]

col_order_daily = ret_cols + lag_cols + cum_cols + ["VIXCLS"] + surp_cols
daily = daily[col_order_daily]

print(f"Dataset 1 (daily)   : {daily.shape}  rows × cols")

# ── 9. Build Dataset 2 — release-date structure ───────────────────────────────
# Rows = union of all release dates present in the surprise panel
rel_dates = surp_cropped.index  # already aligned to [START, END]

# For each release date, we want:
#   - all surp_ values (already in surp_cropped, 0 for non-releases)
#   - ret_ = next-day return log(Close_{t+1}/Close_t)  — tradable
#   - ret_*_lag1 = same-day return log(Close_t/Close_{t-1}) — AR(1) predictor
#   - cum5/10/15 starting at t+1 (i.e., forward from that day)
#   - VIXCLS on that day

ann = surp_cropped.copy()

# Next-day (tradable) returns at release dates
ret_nextday_cropped = ret_nextday.loc[START:END]
ann = ann.join(ret_nextday_cropped.reindex(rel_dates), how="left")

# AR(1): same-day return (lag-1 of next-day return = today's return)
lag1_ann = ret_cropped.copy()
lag1_ann.columns = [c + "_lag1" for c in lag1_ann.columns]
ann = ann.join(lag1_ann.reindex(rel_dates), how="left")

# VIX at release date
ann = ann.join(vix_cropped.reindex(rel_dates), how="left")

# Forward cumulative returns at release date
for H in CUM_HORIZONS:
    ann = ann.join(cum_cropped[H].reindex(rel_dates), how="left")

# Column order for Dataset 2
cum_cols_ann  = [c for c in ann.columns if "_cum" in c]
ret_cols_ann  = [c for c in ann.columns if c.startswith("ret_") and "_cum" not in c and "_lag1" not in c]
lag_cols_ann  = [c for c in ann.columns if c.endswith("_lag1")]

col_order_ann = surp_cols + ret_cols_ann + lag_cols_ann + ["VIXCLS"] + cum_cols_ann
ann = ann[col_order_ann]

print(f"Dataset 2 (announce): {ann.shape}  rows × cols")

# ── 10. Save ──────────────────────────────────────────────────────────────────
daily.to_csv("merged_daily_panel_fedincl_nextday.csv")
ann.to_csv("merged_announcement_panel_fedincl_nextday.csv")
print("\nSaved:")
print("  merged_daily_panel_fedincl_nextday.csv")
print("  merged_announcement_panel_fedincl_nextday.csv")

# Quick sanity check
print("\n--- Daily panel head ---")
print(daily[ret_cols[:3] + ["VIXCLS"] + surp_cols[:3]].head(5).to_string())
print("\n--- Announcement panel head ---")
print(ann[surp_cols[:3] + ret_cols_ann[:3] + lag_cols_ann[:3] + ["VIXCLS"]].head(5).to_string())
