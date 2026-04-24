import os
import pandas as pd
import yfinance as yf

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR   = os.path.join(_SCRIPT_DIR, '..', 'FINAL - Datasets used for final analysis')

data = yf.download(["^GSPC", "^RUT", "XLF", "XLE", "XLK", "XLI", "XLV", "ZT=F", "ZN=F", "EURUSD=X", "CADUSD=X", "JPYUSD=X", "GBPUSD=X", "GC=F"],
                    start="1990-01-01", end="2026-01-01")

print(data)

data.to_csv(os.path.join(_DATA_DIR, "assets_yf_data.csv"))
