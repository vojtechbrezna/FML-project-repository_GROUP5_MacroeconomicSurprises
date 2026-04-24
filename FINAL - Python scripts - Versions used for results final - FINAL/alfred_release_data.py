import os
import requests
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR   = os.path.join(_SCRIPT_DIR, '..', 'FINAL - Datasets used for final analysis')

api_key = "3523293bee6575a9750c8333e279aad4"  # free at fred.stlouisfed.org/docs/api/api_key.html
series_ids = ["A191RL1Q225SBEA"]  # add more IDs here

url = "https://api.stlouisfed.org/fred/series/observations"

dfs = []
for series_id in series_ids:
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "output_type": 4,
        "units": "lin",   # initial release only
        "realtime_start": "1990-07-04",
        "realtime_end": "9999-12-31",
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    obs = r.json()["observations"]
    df = pd.DataFrame(obs)[["date", "value", "realtime_start"]]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["series_id"] = series_id
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
wide = combined.pivot(index="date", columns="series_id", values=["value", "realtime_start"])
wide.columns = [f"{series}_{var}" for var, series in wide.columns]
ordered_cols = [f"{sid}_{var}" for sid in series_ids for var in ["realtime_start", "value"]]
wide = wide[[c for c in ordered_cols if c in wide.columns]]
wide.to_csv(os.path.join(_DATA_DIR, "alfred_releases_mom_chg.csv"))
