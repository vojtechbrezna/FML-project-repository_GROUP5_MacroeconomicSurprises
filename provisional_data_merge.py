import pandas as pd
import numpy as np

from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import acf


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

daily_returns = equity["Close"].pct_change().dropna(how="all")
daily_returns.columns = [f"ret_{t.replace('^', '')}" for t in daily_returns.columns]
# columns: ret_XLE, ret_XLF, ret_XLI, ret_XLK, ret_XLV, ret_GSPC, ret_RUT

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

# ── 5. Build lagged dataset ────────────────────────────────────────────────
ret_cols = [c for c in df.columns if c.startswith("ret_")]
prov_df  = df[ret_cols].copy()                            # returns at t

# Lagged returns: t-1 only
lagged_ret = df[ret_cols].shift(1)
lagged_ret.columns = [f"{c}_lag1" for c in ret_cols]
prov_df = pd.concat([prov_df, lagged_ret], axis=1)

# Event surprises: contemporaneous (lag0) through t-6
for lag in range(0, 7):
    lagged = df[event_abbrev_cols].shift(lag)
    suffix = f"_lag{lag}" if lag > 0 else "_lag0"
    lagged.columns = [f"{c}{suffix}" for c in event_abbrev_cols]
    prov_df = pd.concat([prov_df, lagged], axis=1)

# VIX at t-1
prov_df["vix_lag1"] = df["VIXCLS"].shift(1)

# ── 6. Crop to event date range ────────────────────────────────────────────
first_event = events["date"].min()
last_event  = events["date"].max()
prov_df = prov_df.loc[first_event:last_event]

# Drop rows at the start that don't yet have 6 periods of event history
lag6_cols = [c for c in prov_df.columns if c.endswith("_lag6")]
prov_df = prov_df.dropna(subset=lag6_cols, how="all")

# ── 7. Output ──────────────────────────────────────────────────────────────
print(f"Shape: {prov_df.shape}")
print(f"Date range: {prov_df.index.min().date()} → {prov_df.index.max().date()}")
print(prov_df.head())

prov_df.to_csv("merged_panel.csv")

ret_cols   = [c for c in df.columns if c.startswith("ret_")]

# Check autocorrelation up to 10 lags
for col in ret_cols:
    ac, confint = acf(df[col].dropna(), nlags=5, alpha=0.05)
    print(col)
    for lag in range(1, 6):
        significant = not (confint[lag][0] < 0 < confint[lag][1])
        print(f"  lag {lag}: {ac[lag]:.4f}, significant: {significant}")


