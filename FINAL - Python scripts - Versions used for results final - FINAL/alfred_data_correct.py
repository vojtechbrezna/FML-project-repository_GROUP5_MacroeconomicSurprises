import os
import requests
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR   = os.path.join(_SCRIPT_DIR, '..', 'FINAL - Datasets used for final analysis')

api_key = "3523293bee6575a9750c8333e279aad4"
SERIES_IDS = ["AHETPI"]

# Output mode: "level", "mom", "pct_mom"
OUTPUT_MODE = "pct_mom"

# Frequency: "M" (monthly) or "Q" (quarterly)
FREQ = "Q"


def fetch_series(series_id):
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "realtime_start": "1990-01-01",  # full vintage history → correct release dates
        "realtime_end": "2025-12-31",
    }
    r = requests.get(url, params=params)
    data = r.json()

    if "observations" not in data:
        print(f"  ERROR for {series_id}: {data}")
        return pd.DataFrame()

    df = pd.DataFrame(data["observations"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df["realtime_start"] = pd.to_datetime(df["realtime_start"])
    df["realtime_end"] = pd.to_datetime(df["realtime_end"])

    # Step 1: get all first releases (realtime_start = first time this obs_date appeared)
    first_releases = (
        df.sort_values("realtime_start")
        .groupby("date")
        .first()
        .reset_index()
        [["date", "realtime_start", "value"]]
        .rename(columns={"value": "value_T", "realtime_start": "release_date"})
    )

    # Step 2: for each release date, get best available value for T-1
    records = []
    for _, row in first_releases.iterrows():
        T = row["date"]
        D = row["release_date"]
        T_minus_1 = T - pd.DateOffset(months=3 if FREQ == "Q" else 1)

        candidates = df[
            (df["date"] == T_minus_1) &
            (df["realtime_start"] <= D)
        ]
        if candidates.empty:
            continue
        best_prev = candidates.sort_values("realtime_start").iloc[-1]["value"]

        if pd.isna(row["value_T"]) or pd.isna(best_prev):
            continue

        mom = row["value_T"] - best_prev
        pct_mom = (mom / best_prev * 100) if best_prev != 0 else None

        if OUTPUT_MODE == "level":
            output_value = row["value_T"]
        elif OUTPUT_MODE == "pct_mom":
            output_value = pct_mom
        else:  # "mom"
            output_value = mom

        records.append({
            "release_date": D,
            "date": T,
            "value": output_value,
        })

    result = pd.DataFrame(records)
    result = result[result["date"] >= "1990-01-01"]
    return result


for series_id in SERIES_IDS:
    print(f"Fetching {series_id}...")
    result = fetch_series(series_id)
    if result.empty:
        print(f"  Skipping {series_id} (no data).")
        continue
    result.to_csv(os.path.join(_DATA_DIR, "actual_announced_values_datasets", f"{series_id}_correct_{OUTPUT_MODE}.csv"), index=False)
    print(result.tail(10))
