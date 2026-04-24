# Macro Surprise Predictability and Trading Strategies

Group project studying whether macroeconomic announcement surprises predict short-horizon asset returns. We estimate a broad set of linear and machine-learning models on 14 assets across currencies, equity sectors, and fixed income, then backtest sign-based and quartile-based long/short strategies.

---

## Repository structure

```
├── FINAL - Datasets used for final analysis/          # all input data for final results
├── FINAL - Python scripts - Versions used for results final - FINAL/   # production scripts
├── FINAL - results_final - used for interpretation, tables - FINAL/    # model output files and plots
├── FINAL - strategy_outcomes/                         # backtesting output files and plots
├── outcomes/                                          # live script output directory (mirrors results_final)
├── NOT FINAL  - other datasets that were to decide/   # alternative data sources explored but not used
├── NOT FINAL - Different versions of main pipeline scripts/            # earlier script iterations
├── NOT FINAL - Merged panels - alternative datasets/  # alternative panel constructions
├── NOT FINAL - Provisional(first version) dataset - from Investing.com/
└── NOT FINAL - other versions of strategy backtest python scripts/
```

Folders prefixed **`FINAL -`** contain the data, scripts, and outputs that produced the reported results. Folders prefixed **`NOT FINAL -`** are intermediate or discarded work kept for audit purposes.

> `macro_surprises_final_fedincl.csv` (the raw surprise input) lives in `FINAL - Datasets used for final analysis/`.

---

## Data (`FINAL - Datasets used for final analysis/`)

| File | Description |
|------|-------------|
| `merged_announcement_panel_fedincl_nextday.csv` | Release-date panel: surprise columns + next-day tradable returns + VIX for all 14 assets. Main input to the model selection script. |
| `merged_daily_panel_fedincl_nextday.csv` | Daily panel: all trading days with log returns, AR(1) lags, forward cumulative returns (H = 3, 5, 10, 15), VIX, and surprise columns filled with 0 on non-announcement days. |
| `assets_yf_data.csv` | Raw OHLCV data downloaded from Yahoo Finance (14 tickers, 1990–2026). |
| `rf_return.csv` | Daily risk-free returns used as benchmark in backtesting. |
| `VIX.xlsx` | CBOE VIX daily close series. |
| `Copy of FOMC.xlsx` | FOMC meeting dates, used when building the surprise panel with Fed rate decisions included. |
| `surprises_1.csv` | Raw surprise data (alternative/backup copy). |
| `alfred_releases.csv` | ALFRED vintage release dates and values. |
| `HSN1F_releases.csv` | New Home Sales release history. |
| `actual_announced_values_datasets/` | 16 CSV files with first-release values and release dates for individual macro series (AHETPI, PAYEMS, CPI, CPILFE, UNRATE, UMCSENT, INDPRO, RSAFS, ICSA, GDPC1, BOPGSTB, HSN1F, CES05, A191RL1, DGORDER). |

The intermediate file `macro_surprise_panel_fedincl.csv` (output of `build_surprise_panel.py`) is also written here when the pipeline is re-run.

---

## Scripts (`FINAL - Python scripts - Versions used for results final - FINAL/`)

All scripts use `__file__`-relative paths and can be run from any working directory:

```
python "FINAL - Python scripts - Versions used for results final - FINAL/script_name.py"
```

### Data pipeline (run in order if re-building from scratch)

| Script | Inputs | Outputs | Purpose |
|--------|--------|---------|---------|
| `equity_data.py` | Yahoo Finance API | `assets_yf_data.csv` | Download price data |
| `alfred_data_correct.py` | FRED API | `actual_announced_values_datasets/*.csv` | Download ALFRED first-release values |
| `alfred_release_data.py` | FRED API | `alfred_releases_mom_chg.csv` | Download ALFRED release dates |
| `build_surprise_panel.py` | `macro_surprises_final_fedincl.csv` | `macro_surprise_panel_fedincl.csv` | Pivot raw surprises to date-indexed panel |
| `build_merged_panels.py` | `assets_yf_data.csv`, `VIX.xlsx`, `macro_surprise_panel_fedincl.csv` | `merged_daily_panel_fedincl_nextday.csv`, `merged_announcement_panel_fedincl_nextday.csv` | Assemble analysis-ready panels |

### Analysis scripts (run in order)

| Script | Inputs | Outputs | Purpose |
|--------|--------|---------|---------|
| `descriptive_stats.py` | merged panels, `macro_surprises_final_fedincl.csv` | `outcomes/desc_stats_*.csv` | Summary statistics tables |
| `main_machine_learning_pipeline_final.py` | merged panels, `macro_surprises_final_fedincl.csv` | `outcomes/model_selection_results_*.csv`, `outcomes/y_pred_*.csv`, plots | **Main model estimation** — AR(1), stepwise, Ridge, Lasso, PCR, PLS, Random Forest, LightGBM across 14 assets × 5 return horizons |
| `prediction_eval.py` | `outcomes/y_pred_*.csv`, announcement panel | `outcomes/eval_*.csv` | Pearson correlation and hit-rate evaluation of predictions |
| `strategy_backtesting_gross1_best_oos.py` | `outcomes/model_selection_results_*.csv`, `outcomes/y_pred_*.csv`, daily panel, `rf_return.csv` | `FINAL - strategy_outcomes/` | **Strategy backtesting** — sign-based and quartile-based long/short strategies |

### Sample period splits

| Split | Dates | Role |
|-------|-------|------|
| Tune | up to 2015-12-31 | CV hyperparameter selection (TimeSeriesSplit, K=3) |
| Validation | 2016-01-01 – 2020-12-31 | Expanding-window model evaluation |
| Test | 2021-01-01 onwards | Final holdout / out-of-sample |

### Assets and return horizons

**Assets (14):** CADUSD, EURUSD, GBPUSD, GC (gold), JPYUSD, XLE, XLF, XLI, XLK, XLV, ZN (10Y bond), ZT (2Y bond), GSPC (S&P 500), RUT (Russell 2000)

**Horizons:** spot (1-day next-day), cum3, cum5, cum10, cum15 (forward cumulative log-returns)

### Key methodological details

- **VIX standardisation:** VIXCLS is standardised using the mean and std of daily VIX over the tune period (pre-2016 daily panel), not just announcement days.
- **Surprise standardisation:** each `surp_*` column is divided by its announcement-day std in the tune period, computed from non-zero rows in the raw surprise file (correct zeros are included; structural gaps are excluded).
- **Interaction terms:** surp_j × VIXCLS interactions are appended as features.
- **No AR(1) term in final models** (noar variant): the AR(1) lag is included only as a standalone benchmark, not as a predictor in the estimated models.
- **Checkpoint:** the model loop saves `outcomes/_checkpoint_noar_surpstd.pkl` (~17 MB). If present, the plotting and summary sections load from it directly, skipping the multi-hour estimation.

---

## Results (`FINAL - results_final - used for interpretation, tables - FINAL/`)

- `OOS R2 tables.xlsx` — summary OOS R² tables used in the paper
- `_checkpoint_noar_surpstd.pkl` — serialised model results (required to rerun plots without re-estimating)
- 300+ PNG plots: stepwise variable selection paths, ridge/lasso regularisation paths, PCR/PLS dimension selection, OOS R² bar charts, cumulative OOS squared-error plots, SHAP summaries

## Strategy outcomes (`FINAL - strategy_outcomes/`)

- `strategy1_sign_results.csv` / `strategy2_quartile_results.csv` — per-strategy performance metrics (Sharpe, hit rate, return)
- `benchmark_results.csv` — buy-and-hold benchmarks
- `daily_returns.csv` — full time series of daily strategy P&L
- `filtered_models_list.csv` — models passing the val-R² > 0.0005 filter and best-OOS-R² selection
- `plots/` — cumulative return plots per asset

---

## Dependencies

```
numpy pandas scipy statsmodels scikit-learn lightgbm shap joblib
yfinance requests openpyxl matplotlib ISLP
```

Install with:
```bash
pip install numpy pandas scipy statsmodels scikit-learn lightgbm shap joblib yfinance requests openpyxl matplotlib islp
```

---

## Reproducibility note

Running `linear_models_selection_daily_vix_noar_surpstd.py` from scratch takes several hours due to the expanding-window evaluation across 70 targets. The pre-computed checkpoint in `FINAL - results_final - used for interpretation, tables - FINAL/_checkpoint_noar_surpstd.pkl` is copied to `outcomes/` before re-running the plotting and summary sections to save time.
