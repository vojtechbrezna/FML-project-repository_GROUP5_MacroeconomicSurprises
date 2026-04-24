"""
strategy_backtesting_gross1_best_oos_tc10bps.py
==============================

Variant of strategy_backtesting_gross1.py that additionally keeps only the
highest-OOS-R² model for each target (asset × horizon), after all other filters.

This variant also applies per-trade transaction costs (10 bps) and reports both
gross and net performance.
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration  (edit these to change behaviour)
# ─────────────────────────────────────────────────────────────────────────────
R2_FILTER_THRESHOLD      = 0.0005   # val-R² must exceed this to keep a model
PERCENTILE_THRESHOLD     = 25       # bottom X% → short signal, top X% → long signal
OUTPUT_DIR               = "strategy_outcomes"
TRADING_DAYS_PER_YEAR    = 252
PLOT_SUBSTRATEGIES       = False    # True  → also plot long-only and short-only lines
                                    # False → plot only the full (long+short) strategy
EXCLUDED_MODELS          = {"AR(1)"}  # models to exclude from all analysis
TRANSACTION_COST_BPS     = 10        # one-way cost in basis points (applied on both entry and exit days)
# Manual exclusion of specific (model, target) pairs.
# Target naming follows the CSVs: "ret_{ASSET}" for spot, and "ret_{ASSET}_cumH" for horizons.
MANUAL_EXCLUSIONS        = {
    ("LGB",   "ret_CADUSD"),
    ("RF",    "ret_JPYUSD"),
    ("LGB",   "ret_EURUSD_cum3"),
    ("RF",    "ret_EURUSD_cum3"),
    ("LGB",   "ret_GBPUSD_cum3"),
    ("RF",    "ret_XLV_cum3"),
    ("LGB",   "ret_EURUSD_cum5"),
    ("LGB",   "ret_GBPUSD_cum5"),
    ("LGB",   "ret_ZT_cum5"),
    ("LGB",   "ret_EURUSD_cum10"),
    ("RF",    "ret_EURUSD_cum10"),
    ("LGB",   "ret_XLI_cum10"),
    ("LGB",   "ret_GBPUSD_cum10"),
    ("Lasso", "ret_ZT_cum10"),
    ("Lasso", "ret_CADUSD_cum3"),
}

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
PATH_VAL_R2    = "outcomes/model_selection_results_val_r2.csv"
PATH_OOS_R2    = "outcomes/model_selection_results_oos_r2.csv"
PATH_PRED_TEST = "outcomes/y_pred_test.csv"
PATH_PRED_VAL  = "outcomes/y_pred_val.csv"
PATH_PANEL     = "merged_daily_panel_fedincl_nextday.csv"
PATH_RF        = "rf_return.csv"

OUTPUT_SUBDIR = os.path.join(OUTPUT_DIR, "tc10bps")
os.makedirs(OUTPUT_SUBDIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_SUBDIR, "plots"), exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_horizon(target: str) -> int:
    m = re.search(r"_cum(\d+)$", target)
    return int(m.group(1)) if m else 1


def get_base_col(target: str) -> str:
    """Strip _cumH suffix to get the 1-day return column name."""
    return re.sub(r"_cum\d+$", "", target)


def log_to_simple(r):
    """Convert log return(s) to simple return(s)."""
    return np.expm1(r)


def max_drawdown(cum_series: pd.Series) -> float:
    """Peak-to-trough max drawdown of a cumulative-return series (starting at 1)."""
    roll_max = cum_series.cummax()
    dd = (cum_series - roll_max) / roll_max
    return float(dd.min())


def compute_metrics(daily_r: pd.Series,
                    ann_positions: pd.Series,
                    actual_cumH: pd.Series,
                    rf_series: pd.Series,
                    oos_r2: float) -> dict:
    """
    Parameters
    ----------
    daily_r       : daily portfolio return series over full test window (0 on non-active days)
    ann_positions : positions at announcement-date level (for hit-rate / trade count)
    actual_cumH   : actual cumH returns at announcement dates (for hit-rate)
    rf_series     : daily rf aligned to the same index as daily_r
    oos_r2        : scalar OOS R² from the model selection results
    """
    N          = len(daily_r)
    cum_series = (1.0 + daily_r).cumprod()
    cum_ret    = cum_series.iloc[-1] - 1.0
    ann_ret    = (1.0 + cum_ret) ** (TRADING_DAYS_PER_YEAR / N) - 1.0
    ann_vol    = daily_r.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    rf_ann     = rf_series.mean() * TRADING_DAYS_PER_YEAR
    sharpe     = (ann_ret - rf_ann) / ann_vol if ann_vol > 1e-12 else np.nan
    ann_exc    = ann_ret - rf_ann
    mdd        = max_drawdown(cum_series)

    # hit rate: announcement-level, a trade wins if position * actual_cumH > 0
    active_mask = ann_positions != 0
    n_trades    = int(active_mask.sum())
    if n_trades > 0:
        trade_pnl = ann_positions[active_mask] * actual_cumH.reindex(ann_positions.index[active_mask])
        hit_rate  = float((trade_pnl > 0).sum() / n_trades)
    else:
        hit_rate = np.nan

    return {
        "cum_return":     round(cum_ret,    4),
        "ann_return":     round(ann_ret,    4),
        "ann_excess_ret": round(ann_exc,    4),
        "ann_vol":        round(ann_vol,    4),
        "sharpe":         round(sharpe,     4) if not np.isnan(sharpe) else np.nan,
        "max_drawdown":   round(mdd,        4),
        "oos_r2":         round(oos_r2,     6) if not np.isnan(oos_r2) else np.nan,
        "hit_rate":       round(hit_rate,   4) if not np.isnan(hit_rate) else np.nan,
        "n_trades":       n_trades,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load & filter models
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 1 – Model selection")
print("=" * 65)

val_r2_wide = pd.read_csv(PATH_VAL_R2, index_col=0)
val_r2_wide.index.name = "model"

val_r2_long = (val_r2_wide
               .reset_index()
               .melt(id_vars="model", var_name="target", value_name="val_r2")
               .dropna(subset=["val_r2"]))

selected = (val_r2_long[
                (val_r2_long["val_r2"] > R2_FILTER_THRESHOLD) &
                (~val_r2_long["model"].isin(EXCLUDED_MODELS))
            ]
            .copy()
            .sort_values("val_r2", ascending=False)
            .reset_index(drop=True))

if MANUAL_EXCLUSIONS:
    _keys = pd.MultiIndex.from_frame(selected[["model", "target"]])
    selected = selected.loc[~_keys.isin(MANUAL_EXCLUSIONS)].reset_index(drop=True)

print(f"\nModels with val R² > {R2_FILTER_THRESHOLD}: {len(selected)}")
print(selected.to_string(index=False))
selected.to_csv(os.path.join(OUTPUT_SUBDIR, "filtered_models_list.csv"), index=False)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Load data
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2 – Loading data")
print("=" * 65)

y_pred_test = pd.read_csv(PATH_PRED_TEST, index_col=0, parse_dates=True)
y_pred_test.index.name = "date"

y_pred_val = pd.read_csv(PATH_PRED_VAL, index_col=0, parse_dates=True)
y_pred_val.index.name = "date"

panel = pd.read_csv(PATH_PANEL, index_col=0, parse_dates=True)
panel.index.name = "date"

oos_r2_wide = pd.read_csv(PATH_OOS_R2, index_col=0)
oos_r2_wide.index.name = "model"
oos_r2_long = (oos_r2_wide
               .reset_index()
               .melt(id_vars="model", var_name="target", value_name="oos_r2"))

rf_raw = pd.read_csv(PATH_RF)
rf_raw["date"] = pd.to_datetime(rf_raw["date"].astype(str), format="%Y%m%d")
rf_raw = rf_raw.set_index("date")
rf_raw["rf_daily"] = rf_raw["rf"] / TRADING_DAYS_PER_YEAR

print(f"Predictions test : {y_pred_test.shape}  ({y_pred_test.index.min().date()} – {y_pred_test.index.max().date()})")
print(f"Predictions val  : {y_pred_val.shape}")
print(f"Panel            : {panel.shape}  ({panel.index.min().date()} – {panel.index.max().date()})")

# ─────────────────────────────────────────────────────────────────────────────
# 2b. Keep best model per target by OOS R² (after all other filters)
# ─────────────────────────────────────────────────────────────────────────────
selected_oos = selected.merge(oos_r2_long, on=["model", "target"], how="left")
selected_oos["oos_r2"] = pd.to_numeric(selected_oos["oos_r2"], errors="coerce")
selected_oos = selected_oos.sort_values(["target", "oos_r2"], ascending=[True, False])
selected = (selected_oos
            .drop_duplicates(subset=["target"], keep="first")
            .drop(columns=["oos_r2"])
            .reset_index(drop=True))
print(f"\nBest-by-OOS filter: kept {len(selected)} model/target pairs (1 per target)")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Build test-period panel and helper structures
# ─────────────────────────────────────────────────────────────────────────────
test_start  = y_pred_test.index.min()
panel_test  = panel.loc[test_start:].copy()
panel_dates = panel_test.index                        # all trading days in test window
date_to_idx = {d: i for i, d in enumerate(panel_dates)}

# RF aligned to every panel date, forward-filled then zero-filled
rf_test = rf_raw["rf_daily"].reindex(panel_dates).ffill().fillna(0.0)

# Announcement dates that exist in both predictions and panel
ann_dates = y_pred_test.index.intersection(panel_dates)
print(f"\nAnnouncement dates in test period : {len(ann_dates)}")
print(f"Panel trading days in test window : {len(panel_dates)}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Pre-compute quartile thresholds from validation set (no look-ahead bias)
# ─────────────────────────────────────────────────────────────────────────────
quartile_thresholds: dict[tuple, tuple] = {}
for _, row in selected.iterrows():
    col = f"{row['model']}_{row['target']}"
    if col in y_pred_val.columns:
        vals = y_pred_val[col].dropna().values
        if len(vals) > 0:
            quartile_thresholds[(row["model"], row["target"])] = (
                float(np.percentile(vals, PERCENTILE_THRESHOLD)),
                float(np.percentile(vals, 100 - PERCENTILE_THRESHOLD)),
            )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Build daily P&L for every strategy
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 3 – Building daily P&L series")
print("=" * 65)

STRATEGIES    = ["sign", "quartile"]
SUBSTRATEGIES = ["full", "long", "short"]

# daily_pnl  : label → daily return Series over panel_dates
# ann_pos    : label → position Series over ann_dates (for hit-rate)
daily_pnl_gross: dict[str, pd.Series] = {}
daily_pnl_net:   dict[str, pd.Series] = {}
ann_pos:   dict[str, pd.Series] = {}

skipped = []

for _, row in selected.iterrows():
    model, target = row["model"], row["target"]
    col       = f"{model}_{target}"
    H         = get_horizon(target)
    base_col  = get_base_col(target)

    if col not in y_pred_test.columns:
        skipped.append(col)
        continue
    if base_col not in panel_test.columns:
        skipped.append(f"{col} (base {base_col} missing)")
        continue

    preds = y_pred_test.loc[ann_dates, col].dropna()

    # base 1-day return array for fast indexing (convert log→simple)
    base_ret = log_to_simple(panel_test[base_col].values)  # aligned to panel_dates

    for strat in STRATEGIES:
        if strat == "quartile" and (model, target) not in quartile_thresholds:
            continue
        q_low, q_high = (quartile_thresholds[(model, target)]
                         if strat == "quartile" else (None, None))

        for sub in SUBSTRATEGIES:
            label = f"{col}|{strat}|{sub}"

            pnl_num_arr = np.zeros(len(panel_dates), dtype=float)   # numerator: sum(pos_i * r)
            gross_arr   = np.zeros(len(panel_dates), dtype=float)   # gross exposure: sum(|pos_i|)
            entry_gross_arr = np.zeros(len(panel_dates), dtype=float)  # gross added by new entries at day i
            exit_gross_arr  = np.zeros(len(panel_dates), dtype=float)  # gross removed by exits at day i
            pos_dict: dict = {}

            for ann_date, pred in preds.items():
                # --- determine position ---
                if strat == "sign":
                    if sub == "full":
                        pos = float(np.sign(pred))
                    elif sub == "long":
                        pos = 1.0 if pred > 0 else 0.0
                    else:
                        pos = -1.0 if pred < 0 else 0.0
                else:  # quartile
                    if sub == "full":
                        pos = (1.0 if pred > q_high else
                               -1.0 if pred < q_low else 0.0)
                    elif sub == "long":
                        pos = 1.0 if pred > q_high else 0.0
                    else:
                        pos = -1.0 if pred < q_low else 0.0

                pos_dict[ann_date] = pos
                if pos == 0.0:
                    continue

                # --- earn returns from t+1 over H trading days (inclusive of next day) ---
                start_idx = date_to_idx.get(ann_date)
                if start_idx is None:
                    continue
                start_idx = start_idx + 1
                if start_idx >= len(panel_dates):
                    continue

                end_idx = min(start_idx + H, len(panel_dates))
                entry_gross_arr[start_idx] += abs(pos)
                exit_gross_arr[end_idx - 1] += abs(pos)
                for i in range(start_idx, end_idx):
                    pnl_num_arr[i] += pos * base_ret[i]
                    gross_arr[i]   += abs(pos)

            # gross exposure = 1 on active days
            daily_gross_arr = np.divide(pnl_num_arr, gross_arr,
                                        out=np.zeros_like(pnl_num_arr),
                                        where=gross_arr > 0)
            tc_rate = TRANSACTION_COST_BPS / 10_000.0
            tc_arr = np.divide(entry_gross_arr + exit_gross_arr, gross_arr,
                               out=np.zeros_like(entry_gross_arr),
                               where=gross_arr > 0) * tc_rate
            daily_net_arr = daily_gross_arr - tc_arr

            daily_pnl_gross[label] = pd.Series(daily_gross_arr, index=panel_dates, dtype=float)
            daily_pnl_net[label]   = pd.Series(daily_net_arr,   index=panel_dates, dtype=float)
            ann_pos[label]   = pd.Series(pos_dict, dtype=float)

if skipped:
    print(f"  Skipped {len(skipped)} columns (missing in panel or predictions)")

print(f"  Built {len(daily_pnl_gross)} strategy P&L series")

# Save daily returns matrix
daily_pnl_gross_df = pd.DataFrame(daily_pnl_gross)
daily_pnl_gross_df.index.name = "date"
daily_pnl_gross_df.to_csv(os.path.join(OUTPUT_SUBDIR, "daily_returns_gross.csv"))
daily_pnl_net_df = pd.DataFrame(daily_pnl_net)
daily_pnl_net_df.index.name = "date"
daily_pnl_net_df.to_csv(os.path.join(OUTPUT_SUBDIR, "daily_returns_net.csv"))
print(f"  Saved daily_returns_gross.csv  ({daily_pnl_gross_df.shape})")
print(f"  Saved daily_returns_net.csv    ({daily_pnl_net_df.shape})")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Compute metrics for all strategies
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 4 – Computing metrics")
print("=" * 65)

results_sign     = []
results_quartile = []

for _, row in selected.iterrows():
    model, target = row["model"], row["target"]
    col      = f"{model}_{target}"
    H        = get_horizon(target)

    # OOS R²
    mask_oos = (oos_r2_long["model"] == model) & (oos_r2_long["target"] == target)
    oos_r2_val = float(oos_r2_long.loc[mask_oos, "oos_r2"].values[0]) \
        if mask_oos.any() else np.nan

    # Actual cumH returns at announcement dates (for hit-rate)
    if target in panel_test.columns:
        actual_series = panel_test[target]
        # For spot targets (plain ret_*), the tradable return is the next trading day (t→t+1),
        # which appears in the daily panel at date t+1. Align hit-rate to what is traded.
        if "_cum" not in target:
            actual_series = actual_series.shift(-1)
        actual_cumH_log = actual_series.loc[ann_dates].dropna()
        actual_cumH = pd.Series(log_to_simple(actual_cumH_log.values),
                                index=actual_cumH_log.index, dtype=float)
    else:
        actual_cumH = pd.Series(dtype=float)

    for strat in STRATEGIES:
        for sub in SUBSTRATEGIES:
            label = f"{col}|{strat}|{sub}"
            if label not in daily_pnl_gross:
                continue

            metrics_gross = compute_metrics(
                daily_r      = daily_pnl_gross[label],
                ann_positions= ann_pos[label],
                actual_cumH  = actual_cumH,
                rf_series    = rf_test,
                oos_r2       = oos_r2_val,
            )
            metrics_net = compute_metrics(
                daily_r      = daily_pnl_net[label],
                ann_positions= ann_pos[label],
                actual_cumH  = actual_cumH,
                rf_series    = rf_test,
                oos_r2       = oos_r2_val,
            )
            record = {"model": model, "target": target, "horizon": H, "substrategy": sub,
                      **metrics_gross, **{f"net_{k}": v for k, v in metrics_net.items()}}

            if strat == "sign":
                results_sign.append(record)
            else:
                results_quartile.append(record)

results_sign_df     = pd.DataFrame(results_sign)
results_quartile_df = pd.DataFrame(results_quartile)

results_sign_df.to_csv(os.path.join(OUTPUT_SUBDIR, "strategy1_sign_results.csv"),
                       index=False)
results_quartile_df.to_csv(os.path.join(OUTPUT_SUBDIR, "strategy2_quartile_results.csv"),
                           index=False)
print(f"  Sign strategy rows     : {len(results_sign_df)}")
print(f"  Quartile strategy rows : {len(results_quartile_df)}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Buy-and-Hold benchmark
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 5 – Buy-and-Hold benchmark")
print("=" * 65)

base_assets = sorted({get_base_col(t) for t in selected["target"]})
bh_results  = []

for asset in base_assets:
    if asset not in panel_test.columns:
        continue
    daily_r_log = panel_test[asset].dropna()
    daily_r     = pd.Series(log_to_simple(daily_r_log.values), index=daily_r_log.index, dtype=float)
    rf_aligned  = rf_raw["rf_daily"].reindex(daily_r.index).ffill().fillna(0.0)
    N           = len(daily_r)
    cum_s       = (1.0 + daily_r).cumprod()
    cum_ret     = float(cum_s.iloc[-1] - 1.0)
    ann_ret     = (1.0 + cum_ret) ** (TRADING_DAYS_PER_YEAR / N) - 1.0
    ann_vol     = float(daily_r.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    rf_ann      = float(rf_aligned.mean() * TRADING_DAYS_PER_YEAR)
    sharpe      = (ann_ret - rf_ann) / ann_vol if ann_vol > 1e-12 else np.nan
    bh_results.append({
        "asset":          asset,
        "cum_return":     round(cum_ret,  4),
        "ann_return":     round(ann_ret,  4),
        "ann_excess_ret": round(ann_ret - rf_ann, 4),
        "ann_vol":        round(ann_vol,  4),
        "sharpe":         round(sharpe,   4) if not np.isnan(sharpe) else np.nan,
        "max_drawdown":   round(max_drawdown(cum_s), 4),
        "oos_r2":         np.nan,
        "hit_rate":       np.nan,
        "n_trades":       N,
    })

bh_df = pd.DataFrame(bh_results)
bh_df.to_csv(os.path.join(OUTPUT_SUBDIR, "benchmark_results.csv"), index=False)
print(f"  Benchmark assets computed : {len(bh_df)}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Console summary tables
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("STRATEGY 1 – SIGN STRATEGY RESULTS")
print("=" * 80)
print(results_sign_df.to_string(index=False))

print("\n" + "=" * 80)
print("STRATEGY 2 – QUARTILE STRATEGY RESULTS")
print("=" * 80)
print(results_quartile_df.to_string(index=False))

print("\n" + "=" * 80)
print("BENCHMARK – BUY AND HOLD")
print("=" * 80)
print(bh_df.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# 9. Plots – one PNG per base asset
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 6 – Generating plots")
print("=" * 65)

# colour cycle
COLORS = (list(plt.cm.tab10.colors) + list(plt.cm.Set2.colors))

STRAT_STYLE = {
    "sign|full":      dict(linestyle="-",  linewidth=0.9, alpha=0.85),
    "sign|long":      dict(linestyle="--", linewidth=0.8, alpha=0.75),
    "sign|short":     dict(linestyle=":",  linewidth=0.8, alpha=0.75),
    "quartile|full":  dict(linestyle="-",  linewidth=0.9, alpha=0.85),
    "quartile|long":  dict(linestyle="--", linewidth=0.8, alpha=0.75),
    "quartile|short": dict(linestyle=":",  linewidth=0.8, alpha=0.75),
}

for asset in base_assets:
    fig, ax = plt.subplots(figsize=(16, 7))

    # B&H
    if asset in panel_test.columns:
        bh_simple = log_to_simple(panel_test[asset].fillna(0.0).values)
        bh_cum = pd.Series((1.0 + bh_simple).cumprod(), index=panel_test.index, dtype=float)
        ax.plot(bh_cum.index, bh_cum.values,
                color="black", linewidth=2.2, label="Buy & Hold", zorder=10)

    color_idx = 0
    for label, pnl in daily_pnl_net.items():
        parts = label.split("|")
        if len(parts) != 3:
            continue
        col_name, strat, sub = parts

        # skip long-only / short-only lines if flag is off
        if not PLOT_SUBSTRATEGIES and sub != "full":
            continue

        if get_base_col(col_name.split("_", 1)[1] if "_" in col_name else col_name) != asset:
            # derive base from column: strip model prefix then get base
            # col_name = "{model}_{target}", so target = col_name after first underscore
            try:
                target_part = col_name.split("_", 1)[1]  # might not work for "AR(1)_ret_..."
            except IndexError:
                continue
            if get_base_col(target_part) != asset:
                continue

        cum = (1.0 + pnl).cumprod()
        # build a short display label
        try:
            model_part, target_part = col_name.split("_", 1)
        except ValueError:
            model_part, target_part = col_name, ""
        horizon_str = f"H{get_horizon(target_part)}" if target_part else ""
        disp = f"{model_part} {horizon_str} [{strat}/{sub}]"

        style = STRAT_STYLE.get(f"{strat}|{sub}", {})
        ax.plot(cum.index, cum.values,
                color=COLORS[color_idx % len(COLORS)],
                label=disp, **style)
        color_idx += 1

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.5, zorder=1)
    ax.set_title(f"Cumulative Returns (Net) — {asset}", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Net cumulative return  (1.0 = start)")
    ax.legend(fontsize=6, ncol=3, loc="upper left",
              framealpha=0.6, bbox_to_anchor=(0, 1))
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    plot_path = os.path.join(OUTPUT_SUBDIR, "plots", f"{asset}_cumret.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {plot_path}")

print(f"\nAll outputs written to: {OUTPUT_SUBDIR}/")
print("Done.")
