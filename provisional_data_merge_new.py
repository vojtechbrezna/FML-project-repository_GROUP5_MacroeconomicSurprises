import pandas as pd
import numpy as np


# ── Event ID → abbreviation mapping ───────────────────────────────────────
EVENT_MAP = {
    227:  "nfp",            # Nonfarm Payrolls
    300:  "unemp",          # Unemployment Rate
    8:    "ahe",            # Average Hourly Earnings
    294:  "jobless",        # Initial Jobless Claims
    69:   "cpi",            # CPI
    56:   "core_cpi",       # Core CPI
    61:   "core_pce",       # Core PCE Price Index
    375:  "gdp",            # GDP
    173:  "ism_mfg",        # ISM Manufacturing PMI
    256:  "retail_sales",   # Retail Sales
    161:  "ind_prod",       # Industrial Production
    168:  "fed_rate",       # Fed Interest Rate Decision
    286:  "trade_bal",      # Trade Balance
    238:  "ppi",            # PPI
    86:   "durable_goods",  # Durable Goods Orders
    151:  "housing_starts", # Housing Starts
    48:   "conf_board",     # CB Consumer Confidence
    236:  "philly_fed",     # Philadelphia Fed Manufacturing Index
    234:  "pers_income",    # Personal Income
    1057: "jolts",          # JOLTS Job Openings
}

# ── 1. Load & filter events ────────────────────────────────────────────────
events = pd.read_csv("clean_investing.com_2008-2026.csv")
events["date"] = pd.to_datetime(events["date"])
events = events[events["event_id"].isin(EVENT_MAP)]

# Pivot: one row per date, one column per event with surprise value; 0 on non-event days
events_wide = (
    events
    .groupby(["date", "event_id"])["surprise"]
    .last()                  # handle rare same-day duplicates for the same event
    .unstack("event_id")
    .fillna(0)
)
events_wide.columns = [f"surp_{EVENT_MAP[col]}" for col in events_wide.columns]

# ── 2. Load equity → wide daily returns ───────────────────────────────────
equity = pd.read_csv("equity_index_data.csv", header=[0, 1], index_col=0)
equity = equity.iloc[1:]     # drop the "Date" label row
equity.index = pd.to_datetime(equity.index)
equity.index.name = "date"
equity = equity.astype(float)
equity.columns = pd.MultiIndex.from_tuples(
    [(p.strip(), t.strip()) for p, t in equity.columns],
    names=["price", "ticker"],
)

def clean_ticker(t):
    """Strip Yahoo Finance suffixes: ^GSPC → GSPC, ZN=F → ZN, CADUSD=X → CADUSD."""
    return t.replace("^", "").replace("=F", "").replace("=X", "")

close_prices = equity["Close"].copy()
close_prices.columns = [clean_ticker(t) for t in close_prices.columns]

daily_returns = close_prices.pct_change().dropna(how="all")
daily_returns.columns = [f"ret_{t}" for t in daily_returns.columns]
# columns: ret_XLE, ret_XLF, ret_XLI, ret_XLK, ret_XLV, ret_GSPC, ret_RUT,
#          ret_ZN, ret_ZT, ret_CADUSD, ret_EURUSD, ret_GBPUSD, ret_JPYUSD

# ── 3. Load VIX ────────────────────────────────────────────────────────────
vix = pd.read_excel("VIX.xlsx", sheet_name="Daily, Close")
vix = vix.rename(columns={"observation_date": "date"})
vix["date"] = pd.to_datetime(vix["date"])
vix = vix.set_index("date")[["VIXCLS"]]

# ── 4. Merge on date (left join keeps all trading days) ───────────────────
df = daily_returns.copy()
df = df.merge(events_wide, how="left", left_index=True, right_index=True)
df = df.merge(vix, how="left", left_index=True, right_index=True)

event_abbrev_cols = [f"surp_{v}" for v in EVENT_MAP.values()]
df[event_abbrev_cols] = df[event_abbrev_cols].fillna(0)  # non-event days → 0

# ── 5. Add AR(1) terms for each return series ──────────────────────────────
ret_cols = [c for c in df.columns if c.startswith("ret_")]
for col in ret_cols:
    df[f"{col}_lag1"] = df[col].shift(1)

# ── 6. Clean daily panel ───────────────────────────────────────────────────
prov_df = df.copy()

first_event = events["date"].min()
last_event  = events["date"].max()
prov_df = prov_df.loc[first_event:last_event].dropna(how="all")

print(f"Daily panel shape: {prov_df.shape}")
print(f"Date range: {prov_df.index.min().date()} → {prov_df.index.max().date()}")
prov_df.to_csv("merged_panel_daily.csv")

# ── 7. Announcement-date dataset ──────────────────────────────────────────
TICKERS  = ["XLE", "XLF", "XLI", "XLK", "XLV", "GSPC", "RUT", "ZT", "ZN",
             "CADUSD", "EURUSD", "GBPUSD", "JPYUSD"]
HORIZONS = [5, 10, 15]   # forward windows in trading days

# Rows where at least one surprise is non-zero
surp_cols = [c for c in prov_df.columns if c.startswith("surp_")]
ann_mask  = (prov_df[surp_cols] != 0).any(axis=1)
ann_df    = prov_df[ann_mask].copy()

# Cumulative forward returns: compounded from day 0 (announcement) through day t+h
#   rolling(h+1).sum().shift(-h) aligns the h+1-day log-return sum to the start date
for ticker in TICKERS:
    ret_col = f"ret_{ticker}"
    if ret_col not in daily_returns.columns:
        print(f"Warning: {ret_col} not found in daily_returns, skipping")
        continue
    series  = daily_returns[ret_col]
    log_ret = np.log1p(series)
    for h in HORIZONS:
        cum_log = log_ret.rolling(h + 1).sum().shift(-h)
        ann_df[f"{ret_col}_cum{h}"] = np.expm1(cum_log).reindex(ann_df.index)

# AR(1) terms for cumulative returns (lagged within announcement dates)
cum_ret_cols = [c for c in ann_df.columns if "_cum" in c]
for col in cum_ret_cols:
    ann_df[f"{col}_lag1"] = ann_df[col].shift(1)

print(f"Announcement dataset shape: {ann_df.shape}")
ann_df.to_csv("merged_panel_announcements.csv")

# ── 8. GSPC subset ─────────────────────────────────────────────────────────
gspc_ret_cols = ["ret_GSPC"] + [f"ret_GSPC_cum{h}" for h in HORIZONS]
gspc_df = ann_df[surp_cols + gspc_ret_cols].copy()
print(f"GSPC subset shape: {gspc_df.shape}")
gspc_df.to_csv("merged_panel_announcements_GSPC.csv")
