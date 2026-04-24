"""
Lag-compressed macro-surprise panels.

This script reads the existing `merged_panel.csv` (wide daily panel with returns,
AR(1), VIX lag, and surprise impulses/lag columns) and writes alternative panels
that compress the lag structure of macro surprises.

Goal: drastically reduce the number of `surp_*` predictors while preserving
interpretability as an event-time (distributed-lag) response profile.

Outputs (by default, without modifying any other file):
- `merged_panel_splineK{K}_df{M}.csv`  : natural cubic spline basis over lags 1..K
- `merged_panel_almonK{K}_deg{D}.csv`  : Almon-polynomial (degree D) basis
- `merged_panel_binsK{K}.csv`          : horizon bins (piecewise-constant weights)

All outputs keep contemporaneous surprises `surp_*_lag0` and replace lags 1..K
with a small number of basis features per surprise type.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    # statsmodels dependency; used to build natural cubic regression spline basis
    from patsy import dmatrix
except Exception as _exc:  # pragma: no cover
    dmatrix = None


_SURP_LAG0_RE = re.compile(r"^(surp_.+)_lag0$")


@dataclass(frozen=True)
class BinSpec:
    name: str
    start: int  # inclusive
    end: int  # inclusive


def _read_panel(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError(f"Expected a 'date' column in {path}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df.index.name = "date"
    return df


def _surprise_lag0_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        if _SURP_LAG0_RE.match(c):
            cols.append(c)
    if not cols:
        raise ValueError("No surprise lag0 columns found (expected 'surp_*_lag0').")
    return cols


def _surprise_prefix_from_lag0(col: str) -> str:
    m = _SURP_LAG0_RE.match(col)
    if not m:
        raise ValueError(f"Not a surprise lag0 column: {col}")
    return m.group(1)  # e.g., 'surp_nfp'


def _core_columns(df: pd.DataFrame) -> List[str]:
    """Columns to keep unchanged in all outputs (targets and non-surprise predictors)."""
    ret_cols = [c for c in df.columns if c.startswith("ret_") and "_lag" not in c]
    ar_cols = [c for c in df.columns if c.startswith("ret_") and c.endswith("_lag1")]
    other = [c for c in ["vix_lag1"] if c in df.columns]
    keep = ret_cols + ar_cols + other
    missing_other = [c for c in ["vix_lag1"] if c not in df.columns]
    if missing_other:
        raise ValueError(f"Missing required columns: {missing_other}")
    return keep


def _build_lag_matrix(series: pd.Series, K: int) -> np.ndarray:
    """Return an (n, K) matrix with columns [lag1, lag2, ..., lagK]."""
    n = len(series)
    X = np.empty((n, K), dtype=float)
    for k in range(1, K + 1):
        X[:, k - 1] = series.shift(k).to_numpy()
    return X


def _basis_almon(K: int, degree: int) -> np.ndarray:
    """(K, degree+1) Almon polynomial basis evaluated at k=1..K using scaled k in [0,1]."""
    k = np.arange(1, K + 1, dtype=float)
    x = k / float(K)
    return np.column_stack([x**d for d in range(degree + 1)])


def _basis_natural_spline(K: int, df_spline: int) -> np.ndarray:
    """(K, df_spline) natural cubic regression spline basis on k=1..K."""
    if dmatrix is None:
        raise RuntimeError(
            "patsy is required for spline basis but could not be imported. "
            "Install patsy or use --no-spline."
        )
    k = np.arange(1, K + 1, dtype=float)
    B = dmatrix(f"cr(k, df={df_spline}) - 1", {"k": k}, return_type="dataframe")
    return np.asarray(B)


def _compress_surprises(
    df: pd.DataFrame,
    *,
    K: int,
    basis: np.ndarray,
    basis_colnames: Iterable[str],
    suffix: str,
) -> pd.DataFrame:
    """Create compressed surprise features from lag0 impulses using a (K, M) basis."""
    lag0_cols = _surprise_lag0_columns(df)
    out = {}
    for lag0_col in lag0_cols:
        prefix = _surprise_prefix_from_lag0(lag0_col)
        s0 = df[lag0_col].astype(float)
        out[lag0_col] = s0  # keep contemporaneous
        L = _build_lag_matrix(s0, K)  # (n, K)
        Z = L @ basis  # (n, M)
        for mi, bname in enumerate(basis_colnames):
            out[f"{prefix}_{suffix}{bname}"] = Z[:, mi]
    return pd.DataFrame(out, index=df.index)


def _compress_bins(df: pd.DataFrame, *, K: int, bins: List[BinSpec]) -> pd.DataFrame:
    lag0_cols = _surprise_lag0_columns(df)
    out = {}
    for lag0_col in lag0_cols:
        prefix = _surprise_prefix_from_lag0(lag0_col)
        s0 = df[lag0_col].astype(float)
        out[lag0_col] = s0
        L = _build_lag_matrix(s0, K)  # (n, K)
        for b in bins:
            if b.start < 1 or b.end > K or b.start > b.end:
                raise ValueError(f"Invalid bin {b} for K={K}")
            # convert to 0-based slice
            out[f"{prefix}_bin_{b.name}"] = L[:, (b.start - 1) : b.end].sum(axis=1)
    return pd.DataFrame(out, index=df.index)


def build_panels(
    input_csv: str,
    *,
    K: int,
    spline_df: int,
    almon_degree: int,
    bins: List[BinSpec],
    write_spline: bool = True,
    write_almon: bool = True,
    write_bins: bool = True,
) -> List[Tuple[str, pd.DataFrame]]:
    df = _read_panel(input_csv)
    core = df[_core_columns(df)].copy()

    outputs: List[Tuple[str, pd.DataFrame]] = []

    if write_spline:
        B = _basis_natural_spline(K, spline_df)
        basis_colnames = [f"{i+1}" for i in range(B.shape[1])]
        surp = _compress_surprises(
            df,
            K=K,
            basis=B,
            basis_colnames=basis_colnames,
            suffix="spl",
        )
        out = pd.concat([core, surp], axis=1)
        outputs.append((f"merged_panel_splineK{K}_df{spline_df}.csv", out))

    if write_almon:
        A = _basis_almon(K, almon_degree)
        basis_colnames = [f"{d}" for d in range(A.shape[1])]
        surp = _compress_surprises(
            df,
            K=K,
            basis=A,
            basis_colnames=basis_colnames,
            suffix="alm",
        )
        out = pd.concat([core, surp], axis=1)
        outputs.append((f"merged_panel_almonK{K}_deg{almon_degree}.csv", out))

    if write_bins:
        surp = _compress_bins(df, K=K, bins=bins)
        out = pd.concat([core, surp], axis=1)
        outputs.append((f"merged_panel_binsK{K}.csv", out))

    return outputs


def _default_bins(K: int) -> List[BinSpec]:
    # Transparent horizons; tweak if you prefer weekly buckets.
    return [
        BinSpec("d1", 1, 1),
        BinSpec("d2_5", 2, min(5, K)),
        BinSpec("d6_10", 6, min(10, K)),
        BinSpec("d11_15", 11, min(15, K)),
    ]


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Create lag-compressed macro-surprise panels.")
    ap.add_argument("--input", default="merged_panel.csv", help="Input panel CSV (default: merged_panel.csv)")
    ap.add_argument("--K", type=int, default=15, help="Max post-announcement lag (trading days)")
    ap.add_argument("--spline-df", type=int, default=5, help="Spline basis dimension for lags 1..K")
    ap.add_argument("--almon-degree", type=int, default=3, help="Almon polynomial degree for lags 1..K")
    ap.add_argument("--no-spline", action="store_true", help="Skip spline output")
    ap.add_argument("--no-almon", action="store_true", help="Skip Almon output")
    ap.add_argument("--no-bins", action="store_true", help="Skip binned-horizon output")
    args = ap.parse_args(argv)

    if args.K < 1:
        raise ValueError("--K must be >= 1")
    if args.spline_df < 2:
        raise ValueError("--spline-df must be >= 2")
    if args.almon_degree < 0:
        raise ValueError("--almon-degree must be >= 0")

    outputs = build_panels(
        args.input,
        K=args.K,
        spline_df=args.spline_df,
        almon_degree=args.almon_degree,
        bins=_default_bins(args.K),
        write_spline=not args.no_spline,
        write_almon=not args.no_almon,
        write_bins=not args.no_bins,
    )

    for fname, panel in outputs:
        panel = panel.sort_index()
        panel.to_csv(fname)
        surp_cols = [c for c in panel.columns if c.startswith("surp_")]
        print(f"Wrote {fname}: {panel.shape[0]} rows, {panel.shape[1]} cols ({len(surp_cols)} surp predictors)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())











