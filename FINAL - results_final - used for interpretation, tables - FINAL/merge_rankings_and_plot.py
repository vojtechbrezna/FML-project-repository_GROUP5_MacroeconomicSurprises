"""
Merge feature importance rankings (tree models) with coefficient-derived rankings
(linear models) and produce grouped bar plots per asset class and horizon.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
FI_PATH   = os.path.join(BASE_DIR, "feature_importance_rankings.csv")
COEF_PATH = os.path.join(BASE_DIR, "selected_vars_coefs.csv")
PLOT_DIR  = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
MACRO_VARS = [
    "surp_nfp", "surp_unemp", "surp_corecpi", "surp_gdp", "surp_retsales",
    "surp_ism_pmi", "surp_dgorders", "surp_newhomesales", "surp_consconfid",
    "surp_avghourlyearnings", "surp_cpi", "surp_tradebal", "surp_indprod",
    "surp_fedratedec",
]
MAX_RANK = 14

ASSET_CLASSES = {
    "Fixed Income": ["ZT", "ZN"],
    "Equities":     ["GSPC", "RUT", "XLE", "XLF", "XLI", "XLK", "XLV"],
    "Gold":         ["GC"],
    "Forex":        ["EURUSD", "GBPUSD", "JPYUSD", "CADUSD"],
}
AC_ORDER = ["Fixed Income", "Equities", "Gold", "Forex"]

HORIZONS = ["spot", "cum3", "cum5", "cum10", "cum15"]
MODELS   = ["Fwd (Cp)", "Fwd (CV)", "Bwd (Cp)", "Bwd (CV)", "Lasso", "Ridge", "LGB", "RF"]
RIDGE_METHODS  = {"Ridge"}
SPARSE_METHODS = {"Fwd (Cp)", "Fwd (CV)", "Bwd (Cp)", "Bwd (CV)", "Lasso"}

LABELS = {
    "surp_nfp":               "NFP",
    "surp_unemp":             "Unemp",
    "surp_corecpi":           "Core CPI",
    "surp_gdp":               "GDP",
    "surp_retsales":          "Ret Sales",
    "surp_ism_pmi":           "ISM PMI",
    "surp_dgorders":          "Durable Gds",
    "surp_newhomesales":      "New Homes",
    "surp_consconfid":        "Con Conf",
    "surp_avghourlyearnings": "Avg Hrly Earn",
    "surp_cpi":               "CPI",
    "surp_tradebal":          "Trade Bal",
    "surp_indprod":           "Ind Prod",
    "surp_fedratedec":        "Fed Rate",
}

MODEL_COLORS = {
    "Fwd (Cp)": "#4C72B0",
    "Fwd (CV)": "#1F77B4",
    "Bwd (Cp)": "#55A868",
    "Bwd (CV)": "#2CA02C",
    "Lasso":    "#E377C2",
    "Ridge":    "#C44E52",
    "LGB":      "#8172B2",
    "RF":       "#CCB974",
}

TYPO_MAP = {
    "surp_retsailes":          "surp_retsales",
    "surp_avghourlyearnmings": "surp_avghourlyearnings",
}

RIDGE_THRESH = 1e-6

# ── Helpers ────────────────────────────────────────────────────────────────────

def normalise_var(name: str) -> str:
    return TYPO_MAP.get(name, name)


def parse_target(target: str):
    """Return (asset, horizon) from target strings like ret_CADUSD_cum3 or ret_CADUSD."""
    target = target.strip()
    # strip leading ret_
    inner = re.sub(r"^ret_", "", target)
    cum_match = re.search(r"_(cum\d+)$", inner)
    if cum_match:
        horizon = cum_match.group(1)
        asset   = inner[: cum_match.start()]
    else:
        horizon = "spot"
        asset   = inner
    return asset, horizon


def rank_by_abs(values: dict, max_rank: int = MAX_RANK) -> dict:
    """
    Given {var: combined_coef}, return {var: rank} where rank = max_rank for
    highest |coef| down to 1 for lowest non-zero.  Zero coefs get rank 0.
    """
    nonzero = {v: c for v, c in values.items() if c != 0}
    sorted_vars = sorted(nonzero, key=lambda v: abs(nonzero[v]), reverse=True)
    ranks = {}
    for i, var in enumerate(sorted_vars):
        ranks[var] = max_rank - i          # 14, 13, 12, ...
    for var in values:
        if var not in ranks:
            ranks[var] = 0
    return ranks


# ── Step 1: Linear model rankings ─────────────────────────────────────────────

def build_linear_rankings() -> pd.DataFrame:
    df = pd.read_csv(COEF_PATH)
    df["Variable"] = df["Variable"].apply(normalise_var)

    rows = []
    for (method, target), grp in df.groupby(["Method", "target"]):
        asset, horizon = parse_target(target)

        # Separate base and interaction rows
        base_rows = grp[~grp["Variable"].str.endswith("_x_vix") &
                        grp["Variable"].isin(MACRO_VARS)]
        int_rows  = grp[grp["Variable"].str.endswith("_x_vix")]

        base_coefs = dict(zip(base_rows["Variable"], base_rows["Coef"].astype(float)))
        int_coefs  = {}
        for _, row in int_rows.iterrows():
            base_name = row["Variable"].replace("_x_vix", "")
            base_name = normalise_var(base_name)
            if base_name in MACRO_VARS:
                int_coefs[base_name] = float(row["Coef"])

        # Combined coefficient per macro variable
        combined = {}
        for var in MACRO_VARS:
            c = base_coefs.get(var, 0.0) + int_coefs.get(var, 0.0)
            # Apply threshold
            if method in RIDGE_METHODS:
                if abs(c) <= RIDGE_THRESH:
                    c = 0.0
            else:
                # Sparse methods (subset selection + Lasso): not present → 0
                if var not in base_coefs and var not in int_coefs:
                    c = 0.0
            combined[var] = c

        ranks = rank_by_abs(combined)
        for var in MACRO_VARS:
            rows.append({
                "Asset":   asset,
                "Horizon": horizon,
                "Model":   method,
                "Feature": var,
                "Rank":    ranks[var],
                "MaxRank": MAX_RANK,
            })

    return pd.DataFrame(rows)


# ── Step 2: Tree model rankings ───────────────────────────────────────────────

def build_tree_rankings() -> pd.DataFrame:
    df = pd.read_csv(FI_PATH)
    # Drop VIXCLS — the original data encodes VIXCLS as the top-ranked feature
    # (rank 15 for spot, rank 10/2/1 for cum horizons). Macro variable ranks below
    # it already reflect their true relative importance; no re-ranking needed.
    df = df[df["Feature"] != "VIXCLS"].copy()
    df["Feature"] = df["Feature"].apply(normalise_var)
    df["MaxRank"] = MAX_RANK

    # Keep only known macro vars
    df = df[df["Feature"].isin(MACRO_VARS)]
    return df[["Asset", "Horizon", "Model", "Feature", "Rank", "MaxRank"]]


# ── Step 3: Merge & export per-horizon CSVs ───────────────────────────────────

def build_unified_panel() -> pd.DataFrame:
    linear = build_linear_rankings()
    tree   = build_tree_rankings()
    panel  = pd.concat([linear, tree], ignore_index=True)

    # Fill any missing (asset, horizon, model, feature) with 0
    full_index = pd.MultiIndex.from_product(
        [panel["Asset"].unique(), HORIZONS, MODELS, MACRO_VARS],
        names=["Asset", "Horizon", "Model", "Feature"],
    )
    panel = (panel
             .set_index(["Asset", "Horizon", "Model", "Feature"])
             .reindex(full_index, fill_value=0)
             .reset_index())
    panel["MaxRank"] = MAX_RANK

    for horizon in HORIZONS:
        sub = panel[panel["Horizon"] == horizon]
        sub.to_csv(os.path.join(BASE_DIR, f"rankings_panel_{horizon}.csv"), index=False)
        print(f"  Saved rankings_panel_{horizon}.csv  ({len(sub)} rows)")

    return panel


# ── Step 4 & 5: Asset class aggregation ───────────────────────────────────────

def build_ac_summaries(panel: pd.DataFrame):
    asset_to_class = {a: cls for cls, assets in ASSET_CLASSES.items() for a in assets}
    panel["AssetClass"] = panel["Asset"].map(asset_to_class)
    panel = panel.dropna(subset=["AssetClass"])

    avg = (panel
           .groupby(["AssetClass", "Horizon", "Model", "Feature"])["Rank"]
           .mean()
           .reset_index()
           .rename(columns={"Rank": "AvgRank"}))

    # Normalise by the max avg rank actually achieved by any variable for that
    # (model, asset class, horizon) — not the theoretical maximum of 14.
    max_by_model = (avg
                    .groupby(["AssetClass", "Horizon", "Model"])["AvgRank"]
                    .max()
                    .reset_index()
                    .rename(columns={"AvgRank": "MaxAvgRank"}))

    avg = avg.merge(max_by_model, on=["AssetClass", "Horizon", "Model"])
    avg["AvgNormRank"] = np.where(
        avg["MaxAvgRank"] > 0,
        avg["AvgRank"] / avg["MaxAvgRank"],
        0.0,
    )

    # Per-asset normalization for within-class plots (same logic, individual asset level)
    max_per_asset = (panel
                     .groupby(["Asset", "Horizon", "Model"])["Rank"]
                     .max()
                     .reset_index()
                     .rename(columns={"Rank": "_AssetMax"}))
    panel = panel.merge(max_per_asset, on=["Asset", "Horizon", "Model"])
    panel["NormRankAsset"] = np.where(
        panel["_AssetMax"] > 0,
        panel["Rank"] / panel["_AssetMax"],
        0.0,
    )
    panel = panel.drop(columns=["_AssetMax"])

    return panel, avg, max_by_model


# ── Plotting helpers ───────────────────────────────────────────────────────────

GROUP_SPACING = 1.7   # multiplier on x positions — widens gaps between variable groups

def _bar_plot(ax, data_pivot, ylabel, title, xtick_labels):
    n_vars   = len(data_pivot.index)
    n_models = len(data_pivot.columns)
    x        = np.arange(n_vars) * GROUP_SPACING
    width    = 1 / n_models

    for i, model in enumerate(data_pivot.columns):
        offset = (i - n_models / 2 + 0.5) * width
        vals   = data_pivot[model].values
        ax.bar(x + offset, vals, width=width,
               color=MODEL_COLORS.get(model, f"C{i}"),
               label=model, alpha=0.85, edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=22)
    ax.set_ylabel(ylabel, fontsize=23)
    ax.set_title(title, fontsize=26, fontweight="bold")
    ax.tick_params(axis="y", labelsize=21)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def _legend_patches():
    return [mpatches.Patch(color=MODEL_COLORS[m], label=m) for m in MODELS]


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {os.path.relpath(path, BASE_DIR)}")


def _2x2_fig():
    return plt.subplots(2, 2, figsize=(38, 20))


# ── Plot A: Raw average rank ───────────────────────────────────────────────────

def plot_avg_rank(avg: pd.DataFrame, horizon: str):
    fig, axes = _2x2_fig()
    fig.suptitle(f"Average Variable Rank by Model — {horizon}", fontsize=28, fontweight="bold")
    axes_flat = axes.flatten()

    for idx, ac in enumerate(AC_ORDER):
        sub = avg[(avg["AssetClass"] == ac) & (avg["Horizon"] == horizon)]
        pivot = (sub.pivot_table(index="Feature", columns="Model", values="AvgRank", aggfunc="mean")
                    .reindex(index=MACRO_VARS, columns=MODELS, fill_value=0))
        _bar_plot(axes_flat[idx], pivot, "Average Rank", ac,
                  [LABELS[v] for v in MACRO_VARS])

    fig.legend(handles=_legend_patches(), loc="lower center",
               ncol=len(MODELS), fontsize=20, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save(fig, os.path.join(PLOT_DIR, f"avg_rank_{horizon}.png"))


# ── Plot B: Normalised average rank ───────────────────────────────────────────

def plot_norm_rank(avg: pd.DataFrame, horizon: str):
    fig, axes = _2x2_fig()
    fig.suptitle(f"Normalised Average Rank (Rank / 14) by Model — {horizon}",
                 fontsize=28, fontweight="bold")
    axes_flat = axes.flatten()

    for idx, ac in enumerate(AC_ORDER):
        sub = avg[(avg["AssetClass"] == ac) & (avg["Horizon"] == horizon)]
        pivot = (sub.pivot_table(index="Feature", columns="Model", values="AvgNormRank", aggfunc="mean")
                    .reindex(index=MACRO_VARS, columns=MODELS, fill_value=0))
        _bar_plot(axes_flat[idx], pivot, "Avg Rank / 14", ac,
                  [LABELS[v] for v in MACRO_VARS])
        axes_flat[idx].set_ylim(0, 1)

    fig.legend(handles=_legend_patches(), loc="lower center",
               ncol=len(MODELS), fontsize=20, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save(fig, os.path.join(PLOT_DIR, f"norm_rank_{horizon}.png"))


# ── Plot C: Avg rank normalised by each model's own max obtained rank ──────────

def plot_max_rank(avg: pd.DataFrame, horizon: str):
    fig, axes = _2x2_fig()
    fig.suptitle(
        f"Avg Rank / Model's Max Obtained Rank — {horizon}",
        fontsize=22, fontweight="bold",
    )
    axes_flat = axes.flatten()

    for idx, ac in enumerate(AC_ORDER):
        sub = avg[(avg["AssetClass"] == ac) & (avg["Horizon"] == horizon)]
        pivot = (sub.pivot_table(index="Feature", columns="Model", values="AvgNormRank", aggfunc="mean")
                    .reindex(index=MACRO_VARS, columns=MODELS, fill_value=0))
        _bar_plot(axes_flat[idx], pivot, "Avg Rank / Model Max", ac,
                  [LABELS[v] for v in MACRO_VARS])
        axes_flat[idx].set_ylim(0, 1)

    fig.legend(handles=_legend_patches(), loc="lower center",
               ncol=len(MODELS), fontsize=20, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save(fig, os.path.join(PLOT_DIR, f"max_rank_{horizon}.png"))


# ── Plot D: Within-asset-class ─────────────────────────────────────────────────

def plot_within_ac(panel: pd.DataFrame, horizon: str):
    for ac, assets in ASSET_CLASSES.items():
        ac_safe = ac.replace(" ", "_")
        sub_ac  = panel[(panel["AssetClass"] == ac) & (panel["Horizon"] == horizon)]

        if ac == "Gold":
            fig, ax = plt.subplots(1, 1, figsize=(44, 14))
            fig.suptitle(f"Gold (GC) — {horizon}", fontsize=28, fontweight="bold")
            sub = sub_ac[sub_ac["Asset"] == "GC"]
            pivot = (sub.pivot_table(index="Feature", columns="Model", values="NormRankAsset", aggfunc="mean")
                        .reindex(index=MACRO_VARS, columns=MODELS, fill_value=0))
            _bar_plot(ax, pivot, "Rank / Asset Max", "GC", [LABELS[v] for v in MACRO_VARS])
            ax.set_ylim(0, 1)
            fig.legend(handles=_legend_patches(), loc="lower center",
                       ncol=len(MODELS), fontsize=20, frameon=False,
                       bbox_to_anchor=(0.5, -0.02))
            fig.tight_layout(rect=[0, 0.08, 1, 1])
            _save(fig, os.path.join(PLOT_DIR, f"within_{ac_safe}_{horizon}.png"))
            continue

        n = len(assets)
        if ac == "Equities":
            ncols, nrows = 2, 4
        else:
            ncols = min(n, 2)
            nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(44, 16 * nrows),
                                 squeeze=False)
        fig.suptitle(f"{ac} — individual assets — {horizon}",
                     fontsize=28, fontweight="bold")

        for i, asset in enumerate(assets):
            ax  = axes[i // ncols][i % ncols]
            sub = sub_ac[sub_ac["Asset"] == asset]
            pivot = (sub.pivot_table(index="Feature", columns="Model", values="NormRankAsset", aggfunc="mean")
                        .reindex(index=MACRO_VARS, columns=MODELS, fill_value=0))
            _bar_plot(ax, pivot, "Rank / Asset Max", asset, [LABELS[v] for v in MACRO_VARS])
            ax.set_ylim(0, 1)

        for j in range(n, nrows * ncols):
            axes[j // ncols][j % ncols].set_visible(False)

        fig.legend(handles=_legend_patches(), loc="lower center",
                   ncol=len(MODELS), fontsize=20, frameon=False,
                   bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        _save(fig, os.path.join(PLOT_DIR, f"within_{ac_safe}_{horizon}.png"))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Building unified rankings panel …")
    panel = build_unified_panel()

    print("Computing asset-class summaries …")
    panel, avg, _ = build_ac_summaries(panel)

    print("Generating plots …")
    for horizon in HORIZONS:
        print(f"  horizon = {horizon}")
        plot_avg_rank(avg, horizon)
        plot_norm_rank(avg, horizon)
        plot_max_rank(avg, horizon)
        plot_within_ac(panel, horizon)

    print("Done.")


if __name__ == "__main__":
    main()
