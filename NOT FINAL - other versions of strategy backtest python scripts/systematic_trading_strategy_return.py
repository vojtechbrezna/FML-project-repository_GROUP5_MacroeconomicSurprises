"""
Systematic trading strategy exploiting ML-predicted returns for XLE and XLF.

Three-stage analysis:
  Stage 1 – Baseline: sign strategy (threshold=0) on test set (2021–2025)
  Stage 2 – Grid search on validation set (2016–2020) for flat-zone threshold:
               2a. Prediction-std normalisation: thresh = k * rolling_std(pred, 60d)
               2b. Market-vol normalisation:     thresh = k * rolling_std(actual_ret, 60d)
             k in {0.25, 0.50, 0.75, 1.00, 1.25, 1.50}
  Stage 3 – Best k (by Sharpe on validation) applied to test set

Metrics: Annualised Return, Annualised Sharpe, Max Drawdown, Hit Rate, N Trades
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

warnings.filterwarnings("ignore")

# ── constants ──────────────────────────────────────────────────────────────────
MODELS = ["AR(1)", "Fwd (Cp)", "Fwd (CV)", "Bwd (Cp)", "Bwd (CV)",
          "Ridge", "Lasso", "PCR", "PLS", "RF", "LGB"]
ASSETS = ["ret_XLE", "ret_XLF"]
K_GRID             = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]
MIN_PRED_THRESHOLD = 0.0005   # hard absolute threshold (1bp in decimal return space)
ROLL_WIN = 60          # rolling window for std normalisation (trading days)
ANN = 252              # annualisation factor
OUT_DIR = "outcomes"

# ── helpers ────────────────────────────────────────────────────────────────────
def compute_metrics(strat_ret: pd.Series, signal: pd.Series) -> dict:
    r = strat_ret.dropna()
    if r.empty or r.std() == 0:
        return dict(ann_ret=np.nan, sharpe=np.nan, max_dd=np.nan,
                    hit_rate=np.nan, n_trades=np.nan)
    ann_ret  = r.mean() * ANN
    sharpe   = (r.mean() / r.std()) * np.sqrt(ANN)
    cum      = (1 + r).cumprod()
    max_dd   = ((cum / cum.cummax()) - 1).min()
    # n_trades: number of days with an active (non-zero) position
    n_trades = int((signal != 0).sum())
    # hit rate: fraction of active days where predicted direction matched actual return sign
    active_mask = signal != 0
    correct = strat_ret[active_mask].dropna() > 0
    hit_rate = correct.mean() if len(correct) > 0 else np.nan
    return dict(ann_ret=ann_ret, sharpe=sharpe, max_dd=max_dd,
                hit_rate=hit_rate, n_trades=n_trades)


def run_strategy(pred: pd.Series,
                 actual: pd.Series,
                 threshold) -> tuple:
    """
    Signal: +1 (long) if pred > thresh, -1 (short) if pred < -thresh, else 0 (flat).
    threshold can be a scalar or a pd.Series aligned to pred.index.
    Returns (strategy_returns, signal).
    """
    if isinstance(threshold, pd.Series):
        thresh = threshold.reindex(pred.index).fillna(0)
    else:
        thresh = threshold

    # apply hard floor so economically-zero predictions never generate a signal
    thresh = np.maximum(thresh, MIN_PRED_THRESHOLD)

    sig = np.where(pred >  thresh,  1,
          np.where(pred < -thresh, -1, 0))
    signal = pd.Series(sig, index=pred.index, dtype=float)
    strat_ret = signal * actual.reindex(pred.index)
    return strat_ret, signal


def harmonise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns to the format  {model}_{asset}  using MODELS list.
    Works for both val (AR(1)_ret_XLE) and test (AR(1) benchmark_ret_XLE).
    """
    rename = {}
    for col in df.columns:
        for asset in ASSETS:
            suffix = f"_{asset}"
            if col.endswith(suffix):
                model_raw = col[: -len(suffix)]
                matched = None
                for m in MODELS:
                    if model_raw == m or model_raw.startswith(m):
                        matched = m
                        break
                if matched:
                    rename[col] = f"{matched}_{asset}"
    return df.rename(columns=rename)


# ── load data ──────────────────────────────────────────────────────────────────
print("Loading data ...")

val_preds_raw  = pd.read_csv(os.path.join(OUT_DIR, "y_pred_val.csv"),
                              index_col=0, parse_dates=True)
test_preds_raw = pd.read_csv(os.path.join(OUT_DIR, "y_pred_test.csv"),
                              index_col=0, parse_dates=True)
actual_all     = pd.read_csv("test_set.csv",
                              parse_dates=["date"], dayfirst=True,
                              index_col="date")

val_preds  = harmonise_columns(val_preds_raw)
test_preds = harmonise_columns(test_preds_raw)

val_actual  = actual_all.loc[val_preds.index[0]  : val_preds.index[-1],
                             ["ret_XLE", "ret_XLF", "VIXCLS"]].copy()
test_actual = actual_all.loc[test_preds.index[0] : test_preds.index[-1],
                             ["ret_XLE", "ret_XLF", "VIXCLS"]].copy()

# align to prediction index (inner join on dates)
val_preds,  val_actual  = val_preds.align( val_actual,  join="inner", axis=0)
test_preds, test_actual = test_preds.align(test_actual, join="inner", axis=0)

print(f"  Validation : {val_preds.index[0].date()} - {val_preds.index[-1].date()}"
      f"  ({len(val_preds)} days)")
print(f"  Test       : {test_preds.index[0].date()} - {test_preds.index[-1].date()}"
      f"  ({len(test_preds)} days)")


# ── Stage 1 – Baseline (hard threshold = MIN_PRED_THRESHOLD, test set) ────────
print(f"\nStage 1 - Baseline (hard threshold={MIN_PRED_THRESHOLD}) on test set ...")

rows_baseline = []
cumret_baseline = {}   # {asset: {label: pd.Series}}

for asset in ASSETS:
    cumret_baseline[asset] = {}
    bh = test_actual[asset]
    cumret_baseline[asset]["Buy & Hold"] = (1 + bh).cumprod()

    for model in MODELS:
        col = f"{model}_{asset}"
        if col not in test_preds.columns:
            continue
        pred   = test_preds[col]
        actual = test_actual[asset]
        sr, sig = run_strategy(pred, actual, threshold=MIN_PRED_THRESHOLD)
        m = compute_metrics(sr, sig)
        rows_baseline.append(dict(model=model, asset=asset, **m))
        cumret_baseline[asset][model] = (1 + sr).cumprod()

    # ensemble: average prediction across all 11 models for this asset
    ens_cols = [f"{m}_{asset}" for m in MODELS if f"{m}_{asset}" in test_preds.columns]
    ens_pred = test_preds[ens_cols].mean(axis=1)
    sr_ens, sig_ens = run_strategy(ens_pred, test_actual[asset], threshold=MIN_PRED_THRESHOLD)
    m_ens = compute_metrics(sr_ens, sig_ens)
    rows_baseline.append(dict(model="Ensemble", asset=asset, **m_ens))
    cumret_baseline[asset]["Ensemble"] = (1 + sr_ens).cumprod()

df_baseline = pd.DataFrame(rows_baseline)
df_baseline.to_csv(os.path.join(OUT_DIR, "strategy_baseline.csv"), index=False)
print(df_baseline.to_string(index=False))


# ── Stage 2 – Grid search on validation set ────────────────────────────────────
print("\nStage 2 - Threshold grid on validation set ...")

rows_predstd = []
rows_mkvol   = []

for asset in ASSETS:
    actual_val = val_actual[asset]
    mkvol_roll = actual_val.rolling(ROLL_WIN, min_periods=20).std()

    for model in MODELS:
        col = f"{model}_{asset}"
        if col not in val_preds.columns:
            continue
        pred = val_preds[col]
        pred_std_roll = pred.rolling(ROLL_WIN, min_periods=20).std()
        pred_std_roll = pred_std_roll.where(pred_std_roll > 1e-8, other=np.nan).ffill().fillna(1e-8)

        for k in K_GRID:
            # 2a – prediction-std normalisation
            thresh_predstd = k * pred_std_roll
            sr_ps, sig_ps  = run_strategy(pred, actual_val, thresh_predstd)
            m_ps = compute_metrics(sr_ps, sig_ps)
            rows_predstd.append(dict(model=model, asset=asset, k=k, **m_ps))

            # 2b – market-vol normalisation
            thresh_mkvol = k * mkvol_roll
            sr_mv, sig_mv = run_strategy(pred, actual_val, thresh_mkvol)
            m_mv = compute_metrics(sr_mv, sig_mv)
            rows_mkvol.append(dict(model=model, asset=asset, k=k, **m_mv))

df_grid_predstd = pd.DataFrame(rows_predstd)
df_grid_mkvol   = pd.DataFrame(rows_mkvol)
df_grid_predstd.to_csv(os.path.join(OUT_DIR, "strategy_grid_predstd_val.csv"), index=False)
df_grid_mkvol.to_csv(  os.path.join(OUT_DIR, "strategy_grid_mkvol_val.csv"),   index=False)

# best k per (model, asset) -> highest Sharpe on validation
best_k_predstd = (df_grid_predstd
                  .sort_values("sharpe", ascending=False)
                  .groupby(["model", "asset"])
                  .first()["k"]
                  .to_dict())
best_k_mkvol   = (df_grid_mkvol
                  .sort_values("sharpe", ascending=False)
                  .groupby(["model", "asset"])
                  .first()["k"]
                  .to_dict())

print("  Best k (pred-std normalisation):")
for (m, a), k in sorted(best_k_predstd.items()):
    print(f"    {m:15s}  {a}  k={k}")
print("  Best k (mkt-vol normalisation):")
for (m, a), k in sorted(best_k_mkvol.items()):
    print(f"    {m:15s}  {a}  k={k}")


# ── Stage 3 – Best k on test set ───────────────────────────────────────────────
print("\nStage 3 - Best k applied to test set ...")

rows_tuned = []
cumret_tuned_predstd = {}
cumret_tuned_mkvol   = {}

for asset in ASSETS:
    cumret_tuned_predstd[asset] = {}
    cumret_tuned_mkvol[asset]   = {}
    actual_test = test_actual[asset]
    mkvol_roll_test = actual_test.rolling(ROLL_WIN, min_periods=20).std()
    bh = test_actual[asset]
    cumret_tuned_predstd[asset]["Buy & Hold"] = (1 + bh).cumprod()
    cumret_tuned_mkvol[asset]["Buy & Hold"]   = (1 + bh).cumprod()

    for model in MODELS:
        col = f"{model}_{asset}"
        if col not in test_preds.columns:
            continue
        pred = test_preds[col]
        pred_std_roll_test = pred.rolling(ROLL_WIN, min_periods=20).std()
        pred_std_roll_test = pred_std_roll_test.where(pred_std_roll_test > 1e-8, other=np.nan).ffill().fillna(1e-8)

        # pred-std threshold
        k_ps = best_k_predstd.get((model, asset), 0.5)
        thresh_ps = k_ps * pred_std_roll_test
        sr_ps, sig_ps = run_strategy(pred, actual_test, thresh_ps)
        m_ps = compute_metrics(sr_ps, sig_ps)
        rows_tuned.append(dict(model=model, asset=asset,
                               norm="pred_std", best_k=k_ps, **m_ps))
        cumret_tuned_predstd[asset][model] = (1 + sr_ps).cumprod()

        # mkt-vol threshold
        k_mv = best_k_mkvol.get((model, asset), 0.5)
        thresh_mv = k_mv * mkvol_roll_test
        sr_mv, sig_mv = run_strategy(pred, actual_test, thresh_mv)
        m_mv = compute_metrics(sr_mv, sig_mv)
        rows_tuned.append(dict(model=model, asset=asset,
                               norm="mkt_vol", best_k=k_mv, **m_mv))
        cumret_tuned_mkvol[asset][model] = (1 + sr_mv).cumprod()

df_tuned = pd.DataFrame(rows_tuned)
df_tuned.to_csv(os.path.join(OUT_DIR, "strategy_tuned_test.csv"), index=False)
print(df_tuned.to_string(index=False))


# ── Plots ──────────────────────────────────────────────────────────────────────
print("\nGenerating plots ...")

def _make_cumret_plot(cumret_dict: dict, title: str, filepath: str):
    fig, ax = plt.subplots(figsize=(14, 6))
    labels = list(cumret_dict.keys())
    colors = cm.tab20(np.linspace(0, 1, len(labels)))
    for label, color in zip(labels, colors):
        s = cumret_dict[label]
        lw = 2.2 if label in ("Buy & Hold", "Ensemble") else 1.0
        ls = "--" if label == "Buy & Hold" else "-"
        ax.plot(s.index, s.values, label=label, linewidth=lw, linestyle=ls, color=color)
    ax.set_title(title, fontsize=13)
    ax.set_ylabel("Cumulative return (1 = start)")
    ax.axhline(1, color="black", linewidth=0.6, linestyle=":")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  Saved {filepath}")


for asset in ASSETS:
    tag = asset.replace("ret_", "")

    _make_cumret_plot(
        cumret_baseline[asset],
        f"Baseline strategy (hard threshold={MIN_PRED_THRESHOLD}) - {tag}  [test: 2021-2025]",
        os.path.join(OUT_DIR, f"strategy_cumret_{tag}_baseline.png"))

    _make_cumret_plot(
        cumret_tuned_predstd[asset],
        f"Tuned strategy (pred-std threshold, best k) - {tag}  [test: 2021-2025]",
        os.path.join(OUT_DIR, f"strategy_cumret_{tag}_tuned_predstd.png"))

    _make_cumret_plot(
        cumret_tuned_mkvol[asset],
        f"Tuned strategy (mkt-vol threshold, best k) - {tag}  [test: 2021-2025]",
        os.path.join(OUT_DIR, f"strategy_cumret_{tag}_tuned_mkvol.png"))

print("\nDone. Output files:")
for f in ["strategy_baseline.csv",
          "strategy_grid_predstd_val.csv",
          "strategy_grid_mkvol_val.csv",
          "strategy_tuned_test.csv",
          "strategy_cumret_XLE_baseline.png",
          "strategy_cumret_XLF_baseline.png",
          "strategy_cumret_XLE_tuned_predstd.png",
          "strategy_cumret_XLF_tuned_predstd.png",
          "strategy_cumret_XLE_tuned_mkvol.png",
          "strategy_cumret_XLF_tuned_mkvol.png"]:
    print(f"  outcomes/{f}")
