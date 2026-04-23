import pandas as pd
import yfinance as yf
data = yf.download(["^GSPC", "^RUT", "XLF", "XLE", "XLK", "XLI", "XLV", "ZT=F", "ZN=F", "EURUSD=X", "CADUSD=X", "JPYUSD=X", "GBPUSD=X", "GC=F"], 
                    start="1990-01-01", end="2026-01-01")

print(data)

data.to_csv("assets_yf_data.csv")



