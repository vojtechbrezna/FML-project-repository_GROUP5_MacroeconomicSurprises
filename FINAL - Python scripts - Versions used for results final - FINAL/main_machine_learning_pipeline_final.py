import numpy as np
import pandas as pd
import pickle
import os
from functools import partial
from matplotlib.pyplot import subplots, close
from statsmodels.api import OLS
import sklearn.model_selection as skm
import sklearn.linear_model as skl
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import spearmanr
from ISLP.models import ModelSpec as MS
from ISLP.models import (Stepwise, sklearn_selected, sklearn_selection_path)
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from lightgbm import LGBMRegressor
import shap
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# ── Paths (relative to this script's location) ────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR   = os.path.join(_SCRIPT_DIR, '..', 'FINAL - Datasets used for final analysis')
_OUT_DIR    = os.path.join(_SCRIPT_DIR, '..', 'outcomes')
os.makedirs(_OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
#  INTERACTION TRANSFORMER
# ══════════════════════════════════════════════════════════════════════════════

class SurpVixInteractor(BaseEstimator, TransformerMixin):
    """Appends surp_j * vix interaction columns to an already-standardised matrix.
    Stateless: fit() is a no-op, so it is safe inside sklearn Pipelines."""
    def __init__(self, surp_idx, vix_idx):
        self.surp_idx = surp_idx   # list[int] — column indices for surp_ features
        self.vix_idx  = vix_idx    # int        — column index for VIXCLS
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        vix  = X[:, self.vix_idx : self.vix_idx + 1]  # (n, 1)
        surp = X[:, self.surp_idx]                     # (n, k)
        return np.hstack([X, surp * vix])              # (n, p + k)


class VixPreserveScaler(BaseEstimator, TransformerMixin):
    """StandardScaler that treats VIXCLS and pre-standardised surp columns as identity.
    All other columns (e.g. AR lag) are standardised using their sample statistics."""
    def __init__(self, vix_idx, preserve_idx=None):
        self.vix_idx      = vix_idx
        self.preserve_idx = preserve_idx or []
    def fit(self, X, y=None):
        self._sc = StandardScaler().fit(X)
        for idx in [self.vix_idx] + list(self.preserve_idx):
            self._sc.mean_[idx]  = 0.0
            self._sc.scale_[idx] = 1.0
            self._sc.var_[idx]   = 1.0
        return self
    def transform(self, X):
        return self._sc.transform(X)
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
    def inverse_transform(self, X):
        return self._sc.inverse_transform(X)

# ══════════════════════════════════════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════════════════════════════════════

df = pd.read_csv(os.path.join(_DATA_DIR, "merged_announcement_panel_fedincl_nextday.csv"), index_col=0)
df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
df = df.sort_index()


# ── Pre-standardise VIXCLS using the full daily distribution (tune period) ───
_daily_vix = pd.read_csv(os.path.join(_DATA_DIR, "merged_daily_panel_fedincl_nextday.csv"), index_col=0)
_daily_vix.index = pd.to_datetime(_daily_vix.index)
_vix_tune_daily = _daily_vix.loc[:'2015-12-31', 'VIXCLS'].dropna()
vix_daily_mean  = float(_vix_tune_daily.mean())
vix_daily_std   = float(_vix_tune_daily.std())
df['VIXCLS'] = (df['VIXCLS'] - vix_daily_mean) / vix_daily_std
del _daily_vix, _vix_tune_daily
print(f"VIXCLS pre-standardised using daily series: mean={vix_daily_mean:.4f}, std={vix_daily_std:.4f}")

# ── Pre-standardise surprise columns using announcement-day std (tune period) ─
# Divides each surp_* column by the std of its nonzero announcement values
# in the tune period only (≤2015), computed from the raw surprises CSV.
# This correctly handles genuine zero-surprise days (included) vs structural
# zeros (days with no announcement for that series, excluded because they only
# appear in the merged panel, not in the raw CSV rows).
_surp_raw = pd.read_csv(os.path.join(_DATA_DIR, "macro_surprises_final_fedincl.csv"))
surp_stds = {}
for _col in [c for c in df.columns if c.startswith('surp_')]:
    _series   = _col.replace('surp_', '')
    _date_col = f'releasedate_{_series}'
    if _date_col not in _surp_raw.columns:
        continue
    _dates = pd.to_datetime(_surp_raw[_date_col], dayfirst=True, errors='coerce')
    _vals  = pd.to_numeric(_surp_raw[_col], errors='coerce')
    _mask  = _dates.notna() & _vals.notna() & (_dates.dt.year <= 2015)
    _std   = float(_vals[_mask].std())
    if _std > 0:
        surp_stds[_col] = _std
        df[_col] = df[_col] / _std
del _surp_raw
print(f"Surprise series pre-standardised: {len(surp_stds)} series. Stds: { {k: round(v,4) for k,v in surp_stds.items()} }")

tune_df    = df.loc[:'2015-12-31']                     # Step 1: CV tuning
val_df     = df.loc['2016-01-01':'2020-12-31']         # Step 2: expanding-window validation
test_df    = df.loc['2021-01-01':]                     # Step 3: final holdout
pretest_df = df.loc[:'2020-12-31']                     # for refitting before test evaluation

ret_cols  = [c for c in df.columns if c.startswith('ret_') and '_lag' not in c]
surp_cols = [c for c in df.columns if c.startswith('surp_')]

K    = 3
tscv = TimeSeriesSplit(n_splits=K)

print(f"Tune:  {tune_df.index.min().date()} → {tune_df.index.max().date()}  ({len(tune_df)} obs)")
print(f"Val:   {val_df.index.min().date()} → {val_df.index.max().date()}  ({len(val_df)} obs)")
print(f"Test:  {test_df.index.min().date()} → {test_df.index.max().date()}  ({len(test_df)} obs)")
print(f"Targets: {ret_cols}")
print(f"Predictors per target: AR(1) benchmark only + {len(surp_cols)} surp + 1 VIX (no AR term in models)")

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def nCp(sigma2, estimator, X, Y):
    """Negative Cp statistic (higher = better, for sklearn_selected).
    p+1 counts intercept in the parameter penalty."""
    n, p = X.shape
    Yhat = estimator.predict(X)
    RSS  = np.sum((Y - Yhat) ** 2)
    return -(RSS + 2 * (p) * sigma2) / n

def insample_stats(Y, Yhat, p, sigma2):
    """p = number of selected predictors (intercept added internally as +1)."""
    n   = len(Y)
    TSS = np.sum((Y - Y.mean()) ** 2)
    RSS = np.sum((Y - Yhat) ** 2)
    adj_r2 = 1 - (RSS / (n - p - 1)) / (TSS / (n - 1))
    cp     = RSS / sigma2 + 2 * (p + 1) - n
    bic    = n * np.log(RSS / n) + (p + 1) * np.log(n)
    return adj_r2, cp, bic

def test_mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))

def oos_r2(y_train, y_test, y_pred):
    """Campbell-Thompson OOS R-squared (benchmark = fixed training mean).
    Kept for reference; test evaluation uses expanding_oos_r2 instead."""
    ss_res  = np.sum((y_test - y_pred) ** 2)
    ss_null = np.sum((y_test - y_train.mean()) ** 2)
    return float(1.0 - ss_res / ss_null)

def information_coefficient(y_true, y_pred):
    """Spearman rank correlation between predictions and actuals."""
    corr, _ = spearmanr(y_true, y_pred)
    return float(corr)

def expanding_oos_r2(y_actuals, y_preds, pre_val_Y):
    """OOS R² using expanding training mean as null benchmark at each validation step."""
    running_sum, running_count = pre_val_Y.sum(), len(pre_val_Y)
    null_preds = []
    for i in range(len(y_actuals)):
        null_preds.append(running_sum / running_count)
        running_sum   += y_actuals[i]
        running_count += 1
    null_preds = np.array(null_preds)
    ss_res  = np.sum((y_actuals - y_preds) ** 2)
    ss_null = np.sum((y_actuals - null_preds) ** 2)
    return float(1.0 - ss_res / ss_null)

def expanding_window_eval(builder, val_idx, window_df, target, predictors, spec):
    """Expanding-window OOS evaluation.
    At each date t in val_idx, trains on all data in window_df strictly before t,
    then predicts t. builder: callable(X_tr, Y_tr, X_te, spec) -> float."""
    y_preds, y_acts = [], []
    for t_date in val_idx:
        train_w = window_df.loc[:t_date].iloc[:-1]
        if len(train_w) < 2:
            continue
        X_tr = train_w[predictors].values
        Y_tr = train_w[target].values
        X_te = window_df.loc[[t_date], predictors].values
        y_preds.append(float(builder(X_tr, Y_tr, X_te, spec)))
        y_acts.append(float(window_df.loc[t_date, target]))
    return np.array(y_acts), np.array(y_preds)

def _state_to_D_columns(state, D_columns):
    cols = []
    for term in state:
        variables = getattr(term, 'variables', None)
        if variables is not None:
            for v in variables:
                if v in D_columns:
                    cols.append(v)
        name = getattr(term, 'name', None)
        if isinstance(name, str) and name in D_columns:
            cols.append(name)
    return list(dict.fromkeys(cols))

def _cv_mse_of_vars(vars_list, X_raw, predictor_names, Y, cv,
                    vix_name='VIXCLS', already_scaled=False):
    """CV MSE with per-fold scaler + LinearRegression on the given variable subset.
    X_raw: feature matrix. If already_scaled=True (e.g. X_tune_si), no re-scaling is applied —
    the data is assumed to have been standardised first and interactions appended second.
    Otherwise VixPreserveScaler is used if VIXCLS is present, StandardScaler for all others."""
    if not vars_list:
        pipe = Pipeline([('lr', skl.LinearRegression())])
        X_cv = np.zeros((len(Y), 1))
    else:
        col_idx = [predictor_names.index(v) for v in vars_list]
        X_cv    = X_raw[:, col_idx]
        if already_scaled:
            pipe = Pipeline([('lr', skl.LinearRegression())])
        else:
            vix_sub = vars_list.index(vix_name) if vix_name in vars_list else None
            sc      = VixPreserveScaler(vix_sub) if vix_sub is not None else StandardScaler()
            pipe    = Pipeline([('sc', sc), ('lr', skl.LinearRegression())])
    return float(-skm.cross_validate(
        pipe, X_cv, Y, cv=cv, scoring='neg_mean_squared_error')['test_score'].mean())

# ── Module-level helpers for parallelised stepwise CV fold loops ──────────────
# Must be at module level (not closures) so joblib's loky backend can pickle them.

def _fold_dfs_fn(tr_idx, te_idx, X_tune, X_tune_df, predictors,
                 interactor=None, all_cols=None, vix_idx=None, preserve_idx=None):
    """Standardise one CV fold independently — fit scaler on train, apply to both.
    If interactor is given, append surp×vix interaction columns after standardising.
    vix_idx: column index of VIXCLS — uses VixPreserveScaler to keep daily standardisation.
    preserve_idx: additional column indices to treat as identity (pre-standardised surp columns)."""
    fs  = (VixPreserveScaler(vix_idx, preserve_idx=preserve_idx) if vix_idx is not None else StandardScaler()).fit(X_tune[tr_idx])
    Xtr = fs.transform(X_tune[tr_idx])
    Xte = fs.transform(X_tune[te_idx])
    if interactor is not None:
        Xtr = interactor.transform(Xtr)
        Xte = interactor.transform(Xte)
    cols = all_cols if all_cols is not None else predictors
    return (pd.DataFrame(Xtr, columns=cols, index=X_tune_df.index[tr_idx]),
            pd.DataFrame(Xte, columns=cols, index=X_tune_df.index[te_idx]))

def _fwd_fold_worker(tr_idx, te_idx, X_tune, X_tune_df, predictors,
                     design, Y_tune, n_steps_cv, max_steps,
                     interactor=None, all_cols=None, vix_idx=None, preserve_idx=None):
    """Build the full forward selection path on one CV fold and return per-step MSE."""
    Xf_tr, Xf_te = _fold_dfs_fn(tr_idx, te_idx, X_tune, X_tune_df, predictors,
                                  interactor=interactor, all_cols=all_cols, vix_idx=vix_idx,
                                  preserve_idx=preserve_idx)
    _path = sklearn_selection_path(OLS, Stepwise.fixed_steps(
        design, max_steps, direction='forward'))
    _path.fit(Xf_tr, Y_tune[tr_idx])
    _preds = _path.predict(Xf_te)
    n_s = min(_preds.shape[1], n_steps_cv)
    fe  = np.full((n_steps_cv,), np.nan)
    fe[:n_s] = ((_preds[:, :n_s] - Y_tune[te_idx, None]) ** 2).mean(0)
    return fe

def _bwd_fold_worker(tr_idx, te_idx, X_tune, X_tune_df, predictors,
                     design, Y_tune, n_steps_bwd, min_terms,
                     interactor=None, all_cols=None, vix_idx=None, preserve_idx=None):
    """Build the full backward selection path on one CV fold and return per-step MSE."""
    Xf_tr, Xf_te = _fold_dfs_fn(tr_idx, te_idx, X_tune, X_tune_df, predictors,
                                  interactor=interactor, all_cols=all_cols, vix_idx=vix_idx,
                                  preserve_idx=preserve_idx)
    _path = sklearn_selection_path(OLS, Stepwise.fixed_steps(
        design, min_terms, direction='backward',
        initial_terms=list(design.terms)))
    _path.fit(Xf_tr, Y_tune[tr_idx])
    _preds = _path.predict(Xf_te)
    n_s = min(_preds.shape[1], n_steps_bwd)
    fe  = np.full((n_steps_bwd,), np.nan)
    fe[:n_s] = ((_preds[:, :n_s] - Y_tune[te_idx, None]) ** 2).mean(0)
    return fe

def periodic_expanding_window_eval_builder(builder, val_idx, window_df,
                                           target, predictors, spec,
                                           refit_freq='MS'):
    """Expanding-window eval for builder-based (linear) models.
    Fits once per calendar period on all data before that period,
    then predicts every day in the period with the same frozen model.
    Returns (y_acts, y_preds, pred_dates)."""
    y_preds, y_acts, pred_dates = [], [], []
    val_series = pd.Series(val_idx, index=val_idx)
    for _period, group in val_series.groupby(pd.Grouper(freq=refit_freq)):
        period_dates = group.index
        if len(period_dates) == 0:
            continue
        first_date = period_dates[0]
        train_w = window_df.loc[:first_date].iloc[:-1]
        if len(train_w) < 2:
            continue
        X_tr = train_w[predictors].values
        Y_tr = train_w[target].values
        predict_fn = builder(X_tr, Y_tr, spec)   # fit ONCE per period
        for t_date in period_dates:
            if t_date not in window_df.index:
                continue
            X_te = window_df.loc[[t_date], predictors].values
            y_preds.append(float(predict_fn(X_te)))   # predict only
            y_acts.append(float(window_df.loc[t_date, target]))
            pred_dates.append(t_date)
    return np.array(y_acts), np.array(y_preds), pred_dates

def periodic_expanding_window_eval(model_cls, model_params, val_idx,
                                   window_df, target, predictors,
                                   refit_freq='MS'):
    """Expanding-window eval with periodic (default: monthly) refitting.
    Trains once at the start of each period on all data before that period,
    then predicts every day in the period with the same frozen model.
    Returns (y_acts, y_preds, pred_dates)."""
    y_preds, y_acts, pred_dates = [], [], []
    val_series = pd.Series(val_idx, index=val_idx)
    for _period, group in val_series.groupby(pd.Grouper(freq=refit_freq)):
        period_dates = group.index
        if len(period_dates) == 0:
            continue
        first_date = period_dates[0]
        train_w = window_df.loc[:first_date].iloc[:-1]
        if len(train_w) < 2:
            continue
        X_tr = train_w[predictors].values
        Y_tr = train_w[target].values
        model = model_cls(**model_params)
        model.fit(X_tr, Y_tr)
        for t_date in period_dates:
            if t_date not in window_df.index:
                continue
            X_te = window_df.loc[[t_date], predictors].values
            y_preds.append(float(model.predict(X_te)[0]))
            y_acts.append(float(window_df.loc[t_date, target]))
            pred_dates.append(t_date)
    return np.array(y_acts), np.array(y_preds), pred_dates

# ══════════════════════════════════════════════════════════════════════════════
#  PRE-ALLOCATE SUMMARY FIGURES  (one subplot per target, filled inside loop)
# ══════════════════════════════════════════════════════════════════════════════

# ── Split targets by return horizon ──────────────────────────────────────────
_HORIZONS = {
    'spot':  [c for c in ret_cols if not any(c.endswith(s) for s in ('_cum5', '_cum10', '_cum15'))],
    'cum5':  [c for c in ret_cols if c.endswith('_cum5')],
    'cum10': [c for c in ret_cols if c.endswith('_cum10')],
    'cum15': [c for c in ret_cols if c.endswith('_cum15')],
}

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP — one full model comparison per return target
# ══════════════════════════════════════════════════════════════════════════════

_CHECKPOINT = os.path.join(_OUT_DIR, '_checkpoint_noar_surpstd.pkl')

if os.path.exists(_CHECKPOINT):
    print(f"Loading checkpoint from {_CHECKPOINT} — skipping main loop.")
    with open(_CHECKPOINT, 'rb') as _f:
        _ck = pickle.load(_f)
    all_oos   = _ck['all_oos']
    all_cv    = _ck['all_cv']
    all_insam = _ck['all_insam']
    all_val   = _ck['all_val']
    all_preds = _ck['all_preds']
    all_coefs = _ck['all_coefs']
else:
    all_oos   = {}
    all_cv    = {}
    all_insam = {}
    all_val   = {}
    all_preds = {}   # {target: {phase_method: pd.Series(preds, index=dates)}}
    all_coefs = {}   # {target: pd.DataFrame of selected vars + coefficients}

    for _grp, _grp_cols in _HORIZONS.items():
        if not _grp_cols:
            continue

        n_targets = len(_grp_cols)
        _fkw = dict(squeeze=False)
        fig_fwd,   axes_fwd   = subplots(1, n_targets, figsize=(8 * n_targets, 5), **_fkw)
        fig_bwd,   axes_bwd   = subplots(1, n_targets, figsize=(8 * n_targets, 5), **_fkw)
        fig_rpath, axes_rpath = subplots(1, n_targets, figsize=(8 * n_targets, 5), **_fkw)
        fig_rcv,   axes_rcv   = subplots(1, n_targets, figsize=(7 * n_targets, 5), **_fkw)
        fig_lpath, axes_lpath = subplots(1, n_targets, figsize=(8 * n_targets, 5), **_fkw)
        fig_lcv,   axes_lcv   = subplots(1, n_targets, figsize=(7 * n_targets, 5), **_fkw)
        fig_dim,   axes_dim   = subplots(1, n_targets, figsize=(8 * n_targets, 5), **_fkw)
        fig_oos,   axes_oos   = subplots(1, n_targets, figsize=(10 * n_targets, 5), **_fkw)

        for ti, target in enumerate(_grp_cols):

            print(f"\n{'='*70}")
            print(f"  TARGET: {target}")
            print(f"{'='*70}")

            # AR(1) term is benchmark-only — NOT included in model predictors
            ar_term = f"{target}_lag1"
            has_ar1 = (ar_term in df.columns) and df[ar_term].notna().any()
            predictors = surp_cols + ['VIXCLS']   # no AR term

            # ── Clean rows — separate tune / val / pretest / test ─────────────────────
            full_clean    = df[[target] + predictors].dropna()
            tune_clean    = full_clean.loc[:'2015-12-31']
            val_clean     = full_clean.loc['2016-01-01':'2020-12-31']
            pretest_clean = full_clean.loc[:'2020-12-31']
            test_clean    = full_clean.loc['2021-01-01':]

            # ── AR(1) benchmark data (separate dropna on lag column) ──────────────────
            if has_ar1:
                ar1_full_clean    = df[[target, ar_term]].dropna()
                ar1_tune_clean    = ar1_full_clean.loc[:'2015-12-31']
                ar1_val_clean     = ar1_full_clean.loc['2016-01-01':'2020-12-31']
                ar1_pretest_clean = ar1_full_clean.loc[:'2020-12-31']
                ar1_test_clean    = ar1_full_clean.loc['2021-01-01':]

            Y_tune    = tune_clean[target].values
            X_tune_df = tune_clean[predictors]
            X_tune    = X_tune_df.values
            n_tune    = len(Y_tune)

            Y_val    = val_clean[target].values
            X_val    = val_clean[predictors].values

            Y_pretest = pretest_clean[target].values
            X_pretest = pretest_clean[predictors].values

            Y_test    = test_clean[target].values
            X_test_df = test_clean[predictors]
            X_test    = X_test_df.values

            print(f"  Tune obs: {n_tune}  |  Val obs: {len(Y_val)}  |  Test obs: {len(Y_test)}")

            # ── Tuning-set standardization (VIX column kept at daily-standardised values) ─
            surp_idx_l  = [predictors.index(c) for c in surp_cols if c in predictors]
            vix_idx_l   = predictors.index('VIXCLS')
            scaler      = VixPreserveScaler(vix_idx_l, preserve_idx=surp_idx_l).fit(X_tune)
            X_tune_s    = scaler.transform(X_tune)

            # ── Interaction terms: surp × vix on standardised data ───────────────────
            interactor  = SurpVixInteractor(surp_idx_l, vix_idx_l)
            inter_names = [f"{c}_x_vix" for c in surp_cols if c in predictors]
            predictors_l  = predictors + inter_names        # extended list for linear models
            X_tune_si     = interactor.transform(X_tune_s)  # standardised + interactions
            X_tune_s_df_l = pd.DataFrame(X_tune_si, columns=predictors_l, index=X_tune_df.index)

            # ── AR(1) benchmark — fit on its own tune data ────────────────────────────
            if has_ar1:
                ar1_X_tune = ar1_tune_clean[[ar_term]].values
                ar1_Y_tune = ar1_tune_clean[target].values
                ar1_pipe = Pipeline([('sc', StandardScaler()), ('lr', skl.LinearRegression())])
                ar1_cv_mse = float(-skm.cross_validate(
                    ar1_pipe, ar1_X_tune, ar1_Y_tune,
                    cv=tscv, scoring='neg_mean_squared_error')['test_score'].mean())
                ar1_pipe.fit(ar1_X_tune, ar1_Y_tune)

            # ── sigma2 and ISLP design (tuning data only, extended feature set) ───────
            design           = MS(predictors_l).fit(X_tune_s_df_l)
            X_with_intercept = design.transform(X_tune_s_df_l)
            sigma2           = OLS(Y_tune, X_with_intercept).fit().scale
            neg_Cp           = partial(nCp, sigma2)

            D_tune   = design.fit_transform(X_tune_s_df_l).drop('intercept', axis=1)
            X_no_int = np.asarray(D_tune)

            # ── Per-fold standardisation helper (standardise → interact → ISLP) ───────
            def _fold_dfs(tr_idx, te_idx):
                return _fold_dfs_fn(tr_idx, te_idx, X_tune, X_tune_df, predictors,
                                    interactor=interactor, all_cols=predictors_l,
                                    vix_idx=vix_idx_l, preserve_idx=surp_idx_l)

            # ══════════════════════════════════════════════════════════════════════════
            #  PHASE 1: SUBSET SELECTION — spec selection via CV on tuning data
            # ══════════════════════════════════════════════════════════════════════════

            # -- Forward Cp -----------------------------------------------------------
            fwd_strategy_cp = Stepwise.first_peak(design, direction='forward',
                                                   max_terms=len(design.terms))
            forward_Cp = sklearn_selected(OLS, fwd_strategy_cp, scoring=neg_Cp)
            forward_Cp.fit(X_tune_s_df_l, Y_tune)
            fwd_cp_vars   = [c for c in D_tune.columns if c in forward_Cp.selected_state_]
            fwd_cp_cv_mse = _cv_mse_of_vars(fwd_cp_vars, X_tune_si, predictors_l, Y_tune, tscv, already_scaled=True)

            # -- Forward CV path ------------------------------------------------------
            fwd_strategy_cv = Stepwise.fixed_steps(design, len(design.terms), direction='forward')
            forward_cv_path = sklearn_selection_path(OLS, fwd_strategy_cv)
            forward_cv_path.fit(X_tune_s_df_l, Y_tune)
            Yhat_fwd_in     = forward_cv_path.predict(X_tune_s_df_l)   # (n, n_steps)
            n_steps_cv      = len(forward_cv_path.models_)

            cv_fwd_mse = np.array(
                Parallel(n_jobs=-1, verbose = 10)(
                    delayed(_fwd_fold_worker)(
                        tr_idx, te_idx, X_tune, X_tune_df, predictors,
                        design, Y_tune, n_steps_cv, len(design.terms),
                        interactor=interactor, all_cols=predictors_l,
                        vix_idx=vix_idx_l, preserve_idx=surp_idx_l
                    )
                    for tr_idx, te_idx in tscv.split(Y_tune)
                )
            ).T   # (n_steps, K)

            best_cv_step  = int(np.nanargmin(np.nanmean(cv_fwd_mse, axis=1)))
            best_cv_state = forward_cv_path.models_[best_cv_step][0]
            fwd_cv_vars   = _state_to_D_columns(best_cv_state, D_tune.columns)

            # Validation curve for plot (last 20% of tuning data — for visualisation only)
            _plot_val_cut = int(n_tune * 0.8)
            forward_cv_path.fit(X_tune_s_df_l.iloc[:_plot_val_cut], Y_tune[:_plot_val_cut])
            Yhat_val_fwd = forward_cv_path.predict(X_tune_s_df_l.iloc[_plot_val_cut:])
            val_mse_fwd  = ((Yhat_val_fwd - Y_tune[_plot_val_cut:, None]) ** 2).mean(0)
            forward_cv_path.fit(X_tune_s_df_l, Y_tune)   # refit on full tuning data

            # Forward stepwise plot
            ax = axes_fwd[0, ti]
            insample_mse_fwd = ((Yhat_fwd_in - Y_tune[:, None]) ** 2).mean(0)
            n_sf = insample_mse_fwd.shape[0]
            ax.plot(np.arange(n_sf), insample_mse_fwd, 'k', label='In-sample')
            ax.errorbar(np.arange(n_sf),
                        np.nanmean(cv_fwd_mse, axis=1),
                        np.nanstd(cv_fwd_mse, axis=1) / np.sqrt(K),
                        label='TimeSeriesCV', c='r')
            ax.plot(np.arange(n_sf), val_mse_fwd, c='b', label='Validation (last 20%)')
            ax.axvline(best_cv_step, c='r', ls='--', alpha=0.5,
                       label=f'CV opt ({best_cv_step} steps)')
            ax.set_ylabel('MSE', fontsize=11); ax.set_xlabel('# forward steps', fontsize=11)
            ax.set_title(f'Fwd Stepwise — {target}', fontsize=11)
            ax.set_ylim(bottom=0); ax.legend(fontsize=8)

            print("Forward Subset Selection - Done")
            # -- Backward Cp ----------------------------------------------------------
            bwd_strategy_cp = Stepwise.first_peak(design, direction='backward',
                                                   max_terms=len(design.terms),
                                                   initial_terms=list(design.terms))
            backward_Cp = sklearn_selected(OLS, bwd_strategy_cp, scoring=neg_Cp)
            backward_Cp.fit(X_tune_s_df_l, Y_tune)
            bwd_cp_vars   = [c for c in D_tune.columns if c in backward_Cp.selected_state_]
            bwd_cp_cv_mse = _cv_mse_of_vars(bwd_cp_vars, X_tune_si, predictors_l, Y_tune, tscv, already_scaled=True)

            # -- Backward CV path -----------------------------------------------------
            bwd_strategy_cv  = Stepwise.fixed_steps(design, 0, direction='backward',
                                                     initial_terms=list(design.terms))
            backward_cv_path = sklearn_selection_path(OLS, bwd_strategy_cv)
            backward_cv_path.fit(X_tune_s_df_l, Y_tune)
            Yhat_bwd_in      = backward_cv_path.predict(X_tune_s_df_l)
            n_steps_bwd      = len(backward_cv_path.models_)

            cv_bwd_mse = np.array(
                Parallel(n_jobs=-1, verbose = 10)(
                    delayed(_bwd_fold_worker)(
                        tr_idx, te_idx, X_tune, X_tune_df, predictors,
                        design, Y_tune, n_steps_bwd, 0,
                        interactor=interactor, all_cols=predictors_l,
                        vix_idx=vix_idx_l, preserve_idx=surp_idx_l
                    )
                    for tr_idx, te_idx in tscv.split(Y_tune)
                )
            ).T   # (n_steps, K)

            best_bwd_step  = int(np.nanargmin(np.nanmean(cv_bwd_mse, axis=1)))
            best_bwd_state = backward_cv_path.models_[best_bwd_step][0]
            bwd_cv_vars    = _state_to_D_columns(best_bwd_state, D_tune.columns)

            # Validation curve for backward (last 20% of tuning data — for visualisation only)
            backward_cv_path.fit(X_tune_s_df_l.iloc[:_plot_val_cut], Y_tune[:_plot_val_cut])
            Yhat_val_bwd = backward_cv_path.predict(X_tune_s_df_l.iloc[_plot_val_cut:])
            val_mse_bwd  = ((Yhat_val_bwd - Y_tune[_plot_val_cut:, None]) ** 2).mean(0)
            backward_cv_path.fit(X_tune_s_df_l, Y_tune)   # refit on full tuning data

            print("Backward Subset Selection - Done")
            # Backward stepwise plot
            ax = axes_bwd[0, ti]
            insample_mse_bwd = ((Yhat_bwd_in - Y_tune[:, None]) ** 2).mean(0)
            n_sb = insample_mse_bwd.shape[0]
            ax.plot(np.arange(n_sb), insample_mse_bwd, 'k', label='In-sample')
            ax.errorbar(np.arange(n_sb),
                        np.nanmean(cv_bwd_mse, axis=1),
                        np.nanstd(cv_bwd_mse, axis=1) / np.sqrt(K),
                        label='TimeSeriesCV', c='r')
            ax.plot(np.arange(n_sb), val_mse_bwd, c='b', label='Validation (last 20%)')
            ax.axvline(best_bwd_step, c='r', ls='--', alpha=0.5,
                       label=f'CV opt ({best_bwd_step} removed)')
            ax.set_ylabel('MSE', fontsize=11); ax.set_xlabel('# variables removed', fontsize=11)
            ax.set_title(f'Bwd Stepwise — {target}', fontsize=11)
            ax.set_ylim(bottom=0); ax.legend(fontsize=8)

            # -- In-sample predictions for subset methods (tuning data) ---------------
            def _lr_yhat(vars_list, D):
                if not vars_list:
                    return np.full(len(Y_tune), Y_tune.mean())
                return skl.LinearRegression().fit(D[vars_list], Y_tune).predict(D[vars_list])

            Yhat_fwd_cp   = _lr_yhat(fwd_cp_vars, D_tune)
            Yhat_fwd_cv_t = forward_cv_path.predict(X_tune_s_df_l)[:, best_cv_step]
            Yhat_bwd_cp   = _lr_yhat(bwd_cp_vars, D_tune)
            Yhat_bwd_cv_t = backward_cv_path.predict(X_tune_s_df_l)[:, best_bwd_step]

            # Subset in-sample comparison table
            subset_rows = []
            for name, yh, p_model, cv_val in [
                ('Fwd (Cp)',      Yhat_fwd_cp,   len(fwd_cp_vars),   fwd_cp_cv_mse),
                ('Fwd (CV)',      Yhat_fwd_cv_t, len(fwd_cv_vars),
                                  float(np.nanmean(cv_fwd_mse, axis=1)[best_cv_step])),
                ('Bwd (Cp)',      Yhat_bwd_cp,   len(bwd_cp_vars),   bwd_cp_cv_mse),
                ('Bwd (CV)',      Yhat_bwd_cv_t, len(bwd_cv_vars),
                                  float(np.nanmean(cv_bwd_mse, axis=1)[best_bwd_step])),
            ]:
                adj_r2, cp_val, bic = insample_stats(Y_tune, yh, p_model, sigma2)
                subset_rows.append({'Method': name, 'p': p_model,
                                    'Adj.R2': adj_r2, 'Cp': cp_val, 'BIC': bic, 'CV MSE': cv_val})
            print(f"\n--- Subset selection ({target}) ---")
            print(pd.DataFrame(subset_rows).to_string(index=False, float_format='%.5f'))

            # ══════════════════════════════════════════════════════════════════════════
            #  PHASE 1 CONT.: RIDGE, LASSO, PCR, PLS — tuning data
            # ══════════════════════════════════════════════════════════════════════════

            # RIDGE — Pipeline: standardise → interact → Ridge (fits scaler per CV fold)
            lambdas    = 10 ** np.linspace(8, -6, 100) / Y_tune.std()
            grid_ridge = skm.GridSearchCV(
                Pipeline([('sc', VixPreserveScaler(vix_idx_l, preserve_idx=surp_idx_l)), ('ix', interactor), ('ridge', skl.Ridge())]),
                {'ridge__alpha': lambdas}, cv=tscv, scoring='neg_mean_squared_error',
                n_jobs=-1)
            grid_ridge.fit(X_tune, Y_tune)
            ridge_alpha  = grid_ridge.best_params_['ridge__alpha']
            ridge_cv_mse = -grid_ridge.best_score_

            # Ridge solution-path plot — closed-form (exact, on standardised+interacted data)
            _XtX = X_tune_si.T @ X_tune_si    # computed once
            _XtY = X_tune_si.T @ Y_tune       # computed once
            _eye = np.eye(X_tune_si.shape[1])
            soln_ridge = np.array([
                np.linalg.solve(_XtX + a * _eye, _XtY)
                for a in lambdas
            ])

            soln_ridge_df = pd.DataFrame(soln_ridge, columns=predictors_l, index=np.log(lambdas))
            ax = axes_rpath[0, ti]
            for col in soln_ridge_df.columns:
                ax.plot(soln_ridge_df.index, soln_ridge_df[col], lw=0.6, color='steelblue', alpha=0.5)
            ax.axvline(np.log(ridge_alpha), c='k', ls='--', label=f'CV α={ridge_alpha:.2e}')
            ax.set_xlabel('log(λ)', fontsize=11); ax.set_ylabel('Std. coeff.', fontsize=11)
            ax.set_title(f'Ridge path — {target}', fontsize=11); ax.legend(fontsize=8)

            # Ridge CV error plot
            ridge_cv_res = pd.DataFrame({
                'log_alpha': np.log(grid_ridge.cv_results_['param_ridge__alpha'].data.astype(float)),
                'mean_mse':  -grid_ridge.cv_results_['mean_test_score'],
                'std_mse':    grid_ridge.cv_results_['std_test_score'],
            })
            ax = axes_rcv[0, ti]
            ax.errorbar(ridge_cv_res['log_alpha'], ridge_cv_res['mean_mse'],
                        ridge_cv_res['std_mse'] / np.sqrt(K))
            ax.axvline(np.log(ridge_alpha), c='k', ls='--')
            ax.set_xlabel('log(λ)', fontsize=11); ax.set_ylabel('CV MSE', fontsize=11)
            ax.set_title(f'Ridge CV — {target}', fontsize=11)

            print("Ridge - Done")
            # LASSO — Pipeline: standardise → interact → Lasso (fits scaler per CV fold)
            lasso_alphas = np.logspace(-6, 2, 100)
            grid_lasso   = skm.GridSearchCV(
                Pipeline([('sc', VixPreserveScaler(vix_idx_l, preserve_idx=surp_idx_l)), ('ix', interactor),
                          ('lasso', skl.Lasso(max_iter=10000))]),
                {'lasso__alpha': lasso_alphas}, cv=tscv, scoring='neg_mean_squared_error',
                n_jobs=-1)
            grid_lasso.fit(X_tune, Y_tune)
            lasso_alpha  = grid_lasso.best_params_['lasso__alpha']
            lasso_cv_mse = -grid_lasso.best_score_
            p_lasso_nz   = int((np.abs(grid_lasso.best_estimator_['lasso'].coef_) > 1e-8).sum())

            print("Lasso - Done")
            # Use l1_ratio=1.0 to match the Lasso estimator (pure L1, on std+interacted data)
            alphas_path, soln_lasso, _ = skl.Lasso.path(X_tune_si, Y_tune, alphas=lasso_alphas, l1_ratio=1.0)
            if alphas_path[0] > alphas_path[-1]:
                alphas_path = alphas_path[::-1]
                soln_lasso  = soln_lasso[:, ::-1]
            soln_lasso_df = pd.DataFrame(soln_lasso.T, columns=predictors_l,
                                          index=np.log(alphas_path))
            ax = axes_lpath[0, ti]
            for col in soln_lasso_df.columns:
                ax.plot(soln_lasso_df.index, soln_lasso_df[col], lw=0.6, color='darkorange', alpha=0.5)
            ax.axvline(np.log(lasso_alpha), c='k', ls='--', label=f'CV α={lasso_alpha:.2e}')
            ax.set_xlabel('log(λ)', fontsize=11); ax.set_ylabel('Std. coeff.', fontsize=11)
            ax.set_title(f'Lasso path — {target}', fontsize=11); ax.legend(fontsize=8)

            lasso_cv_plot = pd.DataFrame({
                'log_alpha': np.log(lasso_alphas),
                'mean_mse':  -grid_lasso.cv_results_['mean_test_score'],
                'std_mse':    grid_lasso.cv_results_['std_test_score'],
            })
            ax = axes_lcv[0, ti]
            ax.errorbar(lasso_cv_plot['log_alpha'], lasso_cv_plot['mean_mse'],
                        lasso_cv_plot['std_mse'] / np.sqrt(K))
            ax.axvline(np.log(lasso_alpha), c='k', ls='--')
            ax.set_xlabel('log(λ)', fontsize=11); ax.set_ylabel('CV MSE', fontsize=11)
            ax.set_title(f'Lasso CV — {target}', fontsize=11)


            # Ridge vs Lasso coefficient table
            coef_table = pd.DataFrame({
                'Ridge': grid_ridge.best_estimator_['ridge'].coef_,
                'Lasso': grid_lasso.best_estimator_['lasso'].coef_,
            }, index=predictors_l)
            coef_table['Lasso selected'] = coef_table['Lasso'].abs() > 1e-8
            print(f"\n--- Ridge vs Lasso coefficients ({target}) ---")
            print(coef_table.to_string(float_format='%.5f'))

            # PCR — Pipeline: standardise → interact → PCA → LR (fits scaler per CV fold)
            max_comp = min(50, len(predictors_l), X_tune.shape[0] - 1)
            pipe_pcr = Pipeline([('sc', VixPreserveScaler(vix_idx_l, preserve_idx=surp_idx_l)), ('ix', interactor),
                                  ('pca', PCA()), ('lr', skl.LinearRegression())])
            grid_pcr = skm.GridSearchCV(pipe_pcr, {'pca__n_components': range(1, max_comp + 1)},
                                         cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
            grid_pcr.fit(X_tune, Y_tune)
            pcr_best_n = grid_pcr.best_params_['pca__n_components']
            pcr_cv_mse = -grid_pcr.best_score_

            print("PCR - Done")
            # PLS — Pipeline: standardise → interact → PLS (fits scaler per CV fold)
            max_comp_pls = min(20, len(predictors_l))
            grid_pls = skm.GridSearchCV(
                Pipeline([('sc', VixPreserveScaler(vix_idx_l, preserve_idx=surp_idx_l)), ('ix', interactor),
                          ('pls', PLSRegression(scale=False))]),
                {'pls__n_components': range(1, max_comp_pls + 1)},
                cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
            grid_pls.fit(X_tune, Y_tune)
            pls_best_n = grid_pls.best_params_['pls__n_components']
            pls_cv_mse = -grid_pls.best_score_

            print("PLS - Done")
            # PCR vs PLS CV plot
            pcr_n_vals = list(range(1, max_comp + 1))
            pls_n_vals = list(range(1, max_comp_pls + 1))
            pcr_mean   = -grid_pcr.cv_results_['mean_test_score']
            pcr_std    =  grid_pcr.cv_results_['std_test_score']
            pls_mean   = -grid_pls.cv_results_['mean_test_score']
            pls_std    =  grid_pls.cv_results_['std_test_score']
            ax = axes_dim[0, ti]
            ax.errorbar(pcr_n_vals, pcr_mean, pcr_std / np.sqrt(K), color='blue',  label='PCR')
            ax.errorbar(pls_n_vals, pls_mean, pls_std / np.sqrt(K), color='green', label='PLS')
            ax.axvline(pcr_best_n, color='blue',  ls='--', label=f'PCR best ({pcr_best_n})')
            ax.axvline(pls_best_n, color='green', ls='--', label=f'PLS best ({pls_best_n})')
            ax.set_ylabel('CV MSE', fontsize=11); ax.set_xlabel('# components', fontsize=11)
            ax.set_title(f'PCR vs PLS CV — {target}', fontsize=11); ax.legend(fontsize=8)

            # ══════════════════════════════════════════════════════════════════════════
            #  PHASE 1 RESULTS TABLES (tuning set, ≤ 2015)
            # ══════════════════════════════════════════════════════════════════════════

            if has_ar1:
                Yhat_ar1 = ar1_pipe.predict(ar1_X_tune)
            Yhat_ridge = grid_ridge.best_estimator_.predict(X_tune)
            Yhat_lasso = grid_lasso.best_estimator_.predict(X_tune)
            Yhat_pcr   = grid_pcr.best_estimator_.predict(X_tune)
            Yhat_pls   = grid_pls.best_estimator_.predict(X_tune).ravel()

            # Table 1: CV MSE (tuning set)
            cv_rows = []
            if has_ar1:
                cv_rows.append(('AR(1)', ar1_cv_mse, '1 predictor'))
            cv_rows += [
                ('Fwd (Cp)',         fwd_cp_cv_mse, f'{len(fwd_cp_vars)} pred.'),
                ('Fwd (CV)',         float(np.nanmean(cv_fwd_mse, axis=1)[best_cv_step]),
                                     f'{best_cv_step} pred.'),
                ('Bwd (Cp)',         bwd_cp_cv_mse, f'{len(bwd_cp_vars)} pred.'),
                ('Bwd (CV)',         float(np.nanmean(cv_bwd_mse, axis=1)[best_bwd_step]),
                                     f'{best_bwd_step} removed'),
                ('Ridge',            ridge_cv_mse,  f'alpha={ridge_alpha:.2e}'),
                ('Lasso',            lasso_cv_mse,  f'alpha={lasso_alpha:.2e}, nz={p_lasso_nz}'),
                ('PCR',              pcr_cv_mse,    f'{pcr_best_n} comp.'),
                ('PLS',              pls_cv_mse,    f'{pls_best_n} comp.'),
            ]
            cv_df = pd.DataFrame(cv_rows, columns=['Method', 'CV MSE (tscv)', 'Best param'])
            all_cv[target] = cv_df
            print(f"\n--- Table 1: CV MSE ({target}, tuning ≤2015) ---")
            print(cv_df.to_string(index=False, float_format='%.6f'))

            # Table 2: in-sample stats (tuning set)
            # Note: AR(1) in-sample stats use its own tune sample (may differ in length due to dropna)
            insam_rows = []
            _insam_models = []
            if has_ar1:
                _insam_models.append(('AR(1)', Yhat_ar1, 1, ar1_Y_tune))
            _insam_models += [
                ('Fwd (Cp)',  Yhat_fwd_cp,   len(fwd_cp_vars),  Y_tune),
                ('Fwd (CV)',  Yhat_fwd_cv_t, len(fwd_cv_vars),  Y_tune),
                ('Bwd (Cp)',  Yhat_bwd_cp,   len(bwd_cp_vars),  Y_tune),
                ('Bwd (CV)',  Yhat_bwd_cv_t, len(bwd_cv_vars),  Y_tune),
                ('Ridge',     Yhat_ridge,    len(predictors_l), Y_tune),
                ('Lasso',     Yhat_lasso,    p_lasso_nz,        Y_tune),
                ('PCR',       Yhat_pcr,      pcr_best_n,        Y_tune),
                ('PLS',       Yhat_pls,      pls_best_n,        Y_tune),
            ]
            for name, yh, p_model, y_ref in _insam_models:
                adj_r2, cp_val, bic = insample_stats(y_ref, yh, p_model, sigma2)
                insam_rows.append({'Method': name, 'p': p_model,
                                   'Adj.R2': adj_r2, 'Cp': cp_val, 'BIC': bic})
            insam_df = pd.DataFrame(insam_rows)
            all_insam[target] = insam_df
            print(f"\n--- Table 2: In-sample stats ({target}, tuning ≤2015) ---")
            print(insam_df.to_string(index=False, float_format='%.5f'))

            # ══════════════════════════════════════════════════════════════════════════
            #  SELECTED VARIABLES + COEFFICIENTS (subset methods, Ridge, Lasso)
            # ══════════════════════════════════════════════════════════════════════════

            import statsmodels.api as sm_local
            coef_rows = []
            # Subset selection: OLS with Newey-West HAC SE (maxlags=4)
            for mname_c, vars_list_c in [
                ('Fwd (Cp)', fwd_cp_vars), ('Fwd (CV)', fwd_cv_vars),
                ('Bwd (Cp)', bwd_cp_vars), ('Bwd (CV)', bwd_cv_vars),
            ]:
                if vars_list_c:
                    idx_c  = [predictors_l.index(v) for v in vars_list_c]
                    X_sel  = X_tune_si[:, idx_c]
                    res_ols = sm_local.OLS(Y_tune, sm_local.add_constant(X_sel)).fit()
                    nw      = res_ols.get_robustcov_results(
                        cov_type='HAC', maxlags=4, use_correction=True)
                    for i, v in enumerate(vars_list_c):
                        coef_rows.append({
                            'Method':  mname_c, 'Variable': v,
                            'Coef':    nw.params[i + 1],
                            'HAC SE':  nw.bse[i + 1],
                            't-stat':  nw.tvalues[i + 1],
                            'p-value': nw.pvalues[i + 1],
                        })
            # Ridge and Lasso: coefficients only (no t-stats)
            for v, c in zip(predictors_l, grid_ridge.best_estimator_['ridge'].coef_):
                coef_rows.append({'Method': 'Ridge', 'Variable': v, 'Coef': c,
                                  'HAC SE': float('nan'), 't-stat': float('nan'), 'p-value': float('nan')})
            for v, c in zip(predictors_l, grid_lasso.best_estimator_['lasso'].coef_):
                if abs(c) > 1e-8:
                    coef_rows.append({'Method': 'Lasso', 'Variable': v, 'Coef': c,
                                      'HAC SE': float('nan'), 't-stat': float('nan'), 'p-value': float('nan')})
            all_coefs[target] = pd.DataFrame(coef_rows)

            # ══════════════════════════════════════════════════════════════════════════
            #  RANDOM FOREST + LIGHTGBM — tuning, feature importance, SHAP
            # ══════════════════════════════════════════════════════════════════════════

            rf_param_dist = {
                'n_estimators':     [300, 500, 800],
                'max_depth':        [2, 3, 4],
                'max_features':     ['sqrt', 0.2, 0.3],
                'min_samples_leaf': [15, 30, 50],
                'max_samples':      [0.6, 0.8],
            }
            lgb_param_dist = {
                'n_estimators':      [200, 300, 500],
                'learning_rate':     [0.01, 0.02, 0.05],
                'num_leaves':        [10, 20, 31],
                'min_child_samples': [30, 50, 100],
                'feature_fraction':  [0.2, 0.3, 0.5],
                'bagging_fraction':  [0.7, 0.8],
                'bagging_freq':      [1],
                'reg_lambda':        [0.5, 1.0, 2.0],
            }
            _halving_kw = dict(
                factor=3, cv=tscv, scoring='neg_mean_squared_error',
                random_state=42, n_jobs=-1, refit=True,
                min_resources='exhaust',
            )

            rf_search = HalvingRandomSearchCV(
                RandomForestRegressor(random_state=42, n_jobs=-1),
                rf_param_dist, n_candidates=60, **_halving_kw,
            )
            rf_search.fit(X_tune, Y_tune)
            rf_best   = rf_search.best_params_
            rf_cv_mse = -rf_search.best_score_
            print(f"\n  [RF]  Best tune CV-MSE: {rf_cv_mse:.6f}")
            print(f"  [RF]  Best params:      {rf_best}")

            lgb_search = HalvingRandomSearchCV(
                LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
                lgb_param_dist, n_candidates=60, **_halving_kw,
            )
            lgb_search.fit(X_tune, Y_tune)
            lgb_best   = lgb_search.best_params_
            lgb_cv_mse = -lgb_search.best_score_
            print(f"\n  [LGB] Best tune CV-MSE: {lgb_cv_mse:.6f}")
            print(f"  [LGB] Best params:      {lgb_best}")

            # Append RF/LGB CV MSE to the existing CV table for this target
            all_cv[target] = pd.concat([
                all_cv[target],
                pd.DataFrame([
                    {'Method': 'RF',  'CV MSE (tscv)': rf_cv_mse,  'Best param': str(rf_best)},
                    {'Method': 'LGB', 'CV MSE (tscv)': lgb_cv_mse, 'Best param': str(lgb_best)},
                ])
            ], ignore_index=True)

            # Fit on full tuning data (for feature importance + SHAP)
            rf_tune_model = RandomForestRegressor(**rf_best, random_state=42, n_jobs=-1)
            rf_tune_model.fit(X_tune, Y_tune)
            lgb_tune_model = LGBMRegressor(**lgb_best, random_state=42, n_jobs=-1, verbose=-1)
            lgb_tune_model.fit(X_tune, Y_tune)

            # Feature importance plot (top 20 per model)
            _top_n = 20
            fig_fi, axes_fi = plt.subplots(1, 2, figsize=(16, 6))
            for ax_fi, fi_model, fi_label in [
                (axes_fi[0], rf_tune_model,  'RF'),
                (axes_fi[1], lgb_tune_model, 'LGB'),
            ]:
                imp = pd.Series(fi_model.feature_importances_, index=predictors).nlargest(_top_n)
                ax_fi.barh(imp.index[::-1], imp.values[::-1])
                ax_fi.set_xlabel('Importance', fontsize=10)
                ax_fi.set_title(f'{fi_label} Feature Importance — {target}', fontsize=11)
            fig_fi.tight_layout()
            fig_fi.savefig(os.path.join(_OUT_DIR, f'feature_importance_{target}.png'), dpi=120)
            plt.close(fig_fi)

            # SHAP summary plots (beeswarm, on tuning data)
            for shap_model, shap_name in [(rf_tune_model, 'rf'), (lgb_tune_model, 'lgb')]:
                shap_explainer = shap.TreeExplainer(shap_model)
                shap_vals = shap_explainer.shap_values(X_tune)
                fig_shap = plt.figure(figsize=(10, 6))   # explicit new figure — prevents SHAP from drawing on a shared summary figure
                shap.summary_plot(shap_vals, X_tune, feature_names=predictors, show=False)
                plt.title(f'SHAP Summary — {shap_name.upper()} — {target}')
                plt.tight_layout()
                fig_shap.savefig(os.path.join(_OUT_DIR, f'shap_summary_{shap_name}_{target}.png'), dpi=120,
                                 bbox_inches='tight')
                plt.close(fig_shap)

            # ══════════════════════════════════════════════════════════════════════════
            #  PHASE 2: EXPANDING-WINDOW VALIDATION (2016-2020)
            #  Model specs (variable sets, hyperparameters) are fixed from Phase 1.
            #  Only coefficients are refit at each step of the expanding window.
            # ══════════════════════════════════════════════════════════════════════════

            # _pred_list uses the extended (linear) predictor list for subset/linear builders
            _pred_list = predictors_l

            def _build_ar1(X_tr, Y_tr, spec):
                # AR(1) benchmark: X_tr has exactly one column (ar_term)
                pipe = Pipeline([('sc', StandardScaler()), ('lr', skl.LinearRegression())])
                pipe.fit(X_tr, Y_tr)
                return lambda X_te: pipe.predict(X_te)[0]

            def _build_subset(X_tr, Y_tr, spec):
                # standardise → interact → select columns → OLS
                vl = spec['vars_list']
                if not vl:
                    mean_val = float(Y_tr.mean())
                    return lambda X_te: mean_val
                idx = [_pred_list.index(v) for v in vl]
                sc = VixPreserveScaler(vix_idx_l, preserve_idx=surp_idx_l).fit(X_tr)
                X_tr_si = interactor.transform(sc.transform(X_tr))
                lr = skl.LinearRegression().fit(X_tr_si[:, idx], Y_tr)
                def _predict(X_te):
                    return lr.predict(interactor.transform(sc.transform(X_te))[:, idx])[0]
                return _predict

            def _build_ridge(X_tr, Y_tr, spec):
                pipe = Pipeline([('sc', VixPreserveScaler(vix_idx_l, preserve_idx=surp_idx_l)), ('ix', interactor),
                                 ('ridge', skl.Ridge(alpha=spec['alpha']))])
                pipe.fit(X_tr, Y_tr)
                return lambda X_te: pipe.predict(X_te)[0]

            def _build_lasso(X_tr, Y_tr, spec):
                pipe = Pipeline([('sc', VixPreserveScaler(vix_idx_l, preserve_idx=surp_idx_l)), ('ix', interactor),
                                 ('lasso', skl.Lasso(alpha=spec['alpha'], max_iter=10000))])
                pipe.fit(X_tr, Y_tr)
                return lambda X_te: pipe.predict(X_te)[0]

            def _build_pcr(X_tr, Y_tr, spec):
                pipe = Pipeline([('sc', VixPreserveScaler(vix_idx_l, preserve_idx=surp_idx_l)), ('ix', interactor),
                                 ('pca', PCA(n_components=spec['n_comp'])),
                                 ('lr', skl.LinearRegression())])
                pipe.fit(X_tr, Y_tr)
                return lambda X_te: pipe.predict(X_te)[0]

            def _build_pls(X_tr, Y_tr, spec):
                pipe = Pipeline([('sc', VixPreserveScaler(vix_idx_l, preserve_idx=surp_idx_l)), ('ix', interactor),
                                 ('pls', PLSRegression(n_components=spec['n_comp'], scale=False))])
                pipe.fit(X_tr, Y_tr)
                return lambda X_te: pipe.predict(X_te)[0]

            val_window_df = full_clean.loc[:'2020-12-31']
            methods_val = [
                ('Fwd (Cp)',      _build_subset, {'vars_list': fwd_cp_vars}, val_window_df, predictors),
                ('Fwd (CV)',      _build_subset, {'vars_list': fwd_cv_vars}, val_window_df, predictors),
                ('Bwd (Cp)',      _build_subset, {'vars_list': bwd_cp_vars}, val_window_df, predictors),
                ('Bwd (CV)',      _build_subset, {'vars_list': bwd_cv_vars}, val_window_df, predictors),
                ('Ridge',         _build_ridge,  {'alpha': ridge_alpha},     val_window_df, predictors),
                ('Lasso',         _build_lasso,  {'alpha': lasso_alpha},     val_window_df, predictors),
                ('PCR',           _build_pcr,    {'n_comp': pcr_best_n},     val_window_df, predictors),
                ('PLS',           _build_pls,    {'n_comp': pls_best_n},     val_window_df, predictors),
            ]
            if has_ar1:
                ar1_val_window_df = ar1_full_clean.loc[:'2020-12-31']
                methods_val = [('AR(1)', _build_ar1, {}, ar1_val_window_df, [ar_term])] + methods_val

            val_results = []
            for mname, builder, spec, w_df, pred_list in methods_val:
                y_act, y_pred, pred_dates = periodic_expanding_window_eval_builder(
                    builder, val_clean.index if mname != 'AR(1)' else ar1_val_clean.index,
                    w_df, target, pred_list, spec, refit_freq='MS')
                val_results.append({
                    'Method':     mname,
                    'Val MSE':    test_mse(y_act, y_pred),
                    'Val OOS R2': expanding_oos_r2(y_act, y_pred, Y_tune),
                    'Val IC':     information_coefficient(y_act, y_pred),
                })
                all_preds.setdefault(target, {})[f'val_{mname}'] = pd.Series(
                    y_pred, index=pred_dates)

            # RF + LGB validation (monthly expanding window)
            for model_cls, params, mname in [
                (RandomForestRegressor,
                 {**rf_best, 'random_state': 42, 'n_jobs': -1}, 'RF'),
                (LGBMRegressor,
                 {**lgb_best, 'random_state': 42, 'n_jobs': -1, 'verbose': -1}, 'LGB'),
            ]:
                y_act, y_pred, pred_dates = periodic_expanding_window_eval(
                    model_cls, params, val_clean.index, pretest_clean, target, predictors,
                    refit_freq='MS')
                val_results.append({
                    'Method':     mname,
                    'Val MSE':    test_mse(y_act, y_pred),
                    'Val OOS R2': expanding_oos_r2(y_act, y_pred, Y_tune),
                    'Val IC':     information_coefficient(y_act, y_pred),
                })
                all_preds.setdefault(target, {})[f'val_{mname}'] = pd.Series(
                    y_pred, index=pred_dates)

            val_res_df = pd.DataFrame(val_results)
            all_val[target] = val_res_df
            print(f"\n--- Table 3: Validation OOS ({target}, 2016-2020, monthly expanding window) ---")
            print(val_res_df.to_string(index=False, float_format='%.6f'))

            print("Validation set - Done")
            # ══════════════════════════════════════════════════════════════════════════
            #  PHASE 3: TEST SET EVALUATION (2021+)
            #  Expanding-window evaluation — mirrors Phase 2 exactly.
            #  Specs (variable sets, hyperparameters) are fixed from Phase 1;
            #  only coefficients are refit at each step of the expanding test window.
            # ══════════════════════════════════════════════════════════════════════════

            methods_test = [
                ('Fwd (Cp)',  _build_subset, {'vars_list': fwd_cp_vars}, full_clean,  predictors),
                ('Fwd (CV)',  _build_subset, {'vars_list': fwd_cv_vars}, full_clean,  predictors),
                ('Bwd (Cp)',  _build_subset, {'vars_list': bwd_cp_vars}, full_clean,  predictors),
                ('Bwd (CV)',  _build_subset, {'vars_list': bwd_cv_vars}, full_clean,  predictors),
                ('Ridge',     _build_ridge,  {'alpha': ridge_alpha},     full_clean,  predictors),
                ('Lasso',     _build_lasso,  {'alpha': lasso_alpha},     full_clean,  predictors),
                ('PCR',       _build_pcr,    {'n_comp': pcr_best_n},     full_clean,  predictors),
                ('PLS',       _build_pls,    {'n_comp': pls_best_n},     full_clean,  predictors),
            ]
            if has_ar1:
                methods_test = [('AR(1)', _build_ar1, {}, ar1_full_clean, [ar_term])] + methods_test

            oos_rows = []
            _all_test_preds = []   # collect (mname, y_act, y_pred, dates) for cumulative plot
            for mname, builder, spec, w_df, pred_list in methods_test:
                test_idx = test_clean.index if mname != 'AR(1)' else ar1_test_clean.index
                y_act, y_pred, pred_dates = periodic_expanding_window_eval_builder(
                    builder, test_idx, w_df, target, pred_list, spec, refit_freq='MS')
                oos_rows.append((
                    mname,
                    test_mse(y_act, y_pred),
                    expanding_oos_r2(y_act, y_pred, Y_pretest),
                    information_coefficient(y_act, y_pred),
                ))
                all_preds.setdefault(target, {})[f'test_{mname}'] = pd.Series(
                    y_pred, index=pred_dates)
                _all_test_preds.append((mname, y_act, y_pred, pred_dates))

            # RF + LGB test (monthly expanding window)
            for model_cls, params, mname in [
                (RandomForestRegressor,
                 {**rf_best, 'random_state': 42, 'n_jobs': -1}, 'RF'),
                (LGBMRegressor,
                 {**lgb_best, 'random_state': 42, 'n_jobs': -1, 'verbose': -1}, 'LGB'),
            ]:
                y_act, y_pred, pred_dates = periodic_expanding_window_eval(
                    model_cls, params, test_clean.index, full_clean, target, predictors,
                    refit_freq='MS')
                oos_rows.append((
                    mname,
                    test_mse(y_act, y_pred),
                    expanding_oos_r2(y_act, y_pred, Y_pretest),
                    information_coefficient(y_act, y_pred),
                ))
                all_preds.setdefault(target, {})[f'test_{mname}'] = pd.Series(
                    y_pred, index=pred_dates)
                _all_test_preds.append((mname, y_act, y_pred, pred_dates))

            oos_df = pd.DataFrame(oos_rows, columns=['Method', 'Test MSE', 'OOS R2', 'IC'])
            all_oos[target] = oos_df
            print(f"\n--- Table 4: Out-of-sample test ({target}, 2021+) ---")
            print(oos_df.to_string(index=False, float_format='%.6f'))

            # Cumulative excess squared error plot (test set, model vs naive benchmark)
            fig_cum, ax_cum = plt.subplots(figsize=(12, 5))
            for cum_mname, cum_y_act, cum_y_pred, cum_dates in _all_test_preds:
                _running_sum   = Y_pretest.sum()
                _running_count = len(Y_pretest)
                _naive = []
                for _y in cum_y_act:
                    _naive.append(_running_sum / _running_count)
                    _running_sum   += _y
                    _running_count += 1
                _naive_arr = np.array(_naive)
                cum_excess = (np.cumsum((cum_y_act - cum_y_pred) ** 2) -
                              np.cumsum((cum_y_act - _naive_arr) ** 2))
                ax_cum.plot(cum_dates, cum_excess, label=cum_mname, linewidth=0.9)
            ax_cum.axhline(0, color='black', linewidth=0.8, linestyle='--')
            ax_cum.set_title(f'Cumulative Excess Squared Error vs Naive — {target}', fontsize=11)
            ax_cum.set_ylabel('Cumulative (model² − naive²)', fontsize=10)
            ax_cum.legend(fontsize=6, ncol=2)
            fig_cum.tight_layout()
            fig_cum.savefig(os.path.join(_OUT_DIR, f'cumulative_oos_{target}.png'), dpi=120)
            plt.close(fig_cum)

            # OOS R² bar chart (test set)
            ax = axes_oos[0, ti]
            bar_colors = ['gold' if v == oos_df['OOS R2'].max() else 'steelblue'
                          for v in oos_df['OOS R2']]
            bars = ax.bar(oos_df['Method'], oos_df['OOS R2'], color=bar_colors)
            ax.bar_label(bars, fmt='%.4f', fontsize=7, padding=2)
            ax.set_ylabel('OOS R²', fontsize=11)
            ax.set_title(f'OOS R² (expanding hist. mean benchmark) — {target}', fontsize=11)
            ax.set_xticklabels(oos_df['Method'], rotation=35, ha='right', fontsize=8)
            _r2_min = oos_df['OOS R2'].min(); _r2_max = oos_df['OOS R2'].max()
            ax.set_ylim(_r2_min - 0.1 * abs(_r2_min) - 0.001,
                        _r2_max + 0.1 * abs(_r2_max) + 0.001)

        # ══════════════════════════════════════════════════════════════════════════════
        #  SAVE SUMMARY FIGURES
        # ══════════════════════════════════════════════════════════════════════════════

        _summary_plots = [
            (fig_fwd,   os.path.join(_OUT_DIR, f'fwd_stepwise_summary_{_grp}.png')),
            (fig_bwd,   os.path.join(_OUT_DIR, f'bwd_stepwise_summary_{_grp}.png')),
            (fig_rpath, os.path.join(_OUT_DIR, f'ridge_path_summary_{_grp}.png')),
            (fig_rcv,   os.path.join(_OUT_DIR, f'ridge_cv_summary_{_grp}.png')),
            (fig_lpath, os.path.join(_OUT_DIR, f'lasso_path_summary_{_grp}.png')),
            (fig_lcv,   os.path.join(_OUT_DIR, f'lasso_cv_summary_{_grp}.png')),
            (fig_dim,   os.path.join(_OUT_DIR, f'pcr_pls_summary_{_grp}.png')),
            (fig_oos,   os.path.join(_OUT_DIR, f'oos_r2_summary_{_grp}.png')),
        ]
        for fig, fname in _summary_plots:
            fig.tight_layout()
            fig.savefig(fname, dpi=120)
            close(fig)
        print("\nSaved:", [f for _, f in _summary_plots])

    # Save checkpoint so plotting can be rerun without recomputing
    with open(_CHECKPOINT, 'wb') as _f:
        pickle.dump({'all_oos': all_oos, 'all_cv': all_cv, 'all_insam': all_insam,
                     'all_val': all_val, 'all_preds': all_preds, 'all_coefs': all_coefs}, _f)
    print(f"Checkpoint saved to {_CHECKPOINT}")

# ══════════════════════════════════════════════════════════════════════════════
#  CROSS-ASSET SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

summary_oos_mse = pd.concat(
    {t: r.set_index('Method')['Test MSE'] for t, r in all_oos.items()}, axis=1)
summary_oos_r2 = pd.concat(
    {t: r.set_index('Method')['OOS R2'] for t, r in all_oos.items()}, axis=1)
summary_oos_ic = pd.concat(
    {t: r.set_index('Method')['IC'] for t, r in all_oos.items()}, axis=1)
summary_val_mse = pd.concat(
    {t: r.set_index('Method')['Val MSE'] for t, r in all_val.items()}, axis=1)
summary_val_r2 = pd.concat(
    {t: r.set_index('Method')['Val OOS R2'] for t, r in all_val.items()}, axis=1)
summary_val_ic = pd.concat(
    {t: r.set_index('Method')['Val IC'] for t, r in all_val.items()}, axis=1)
summary_cv = pd.concat(
    {t: r.set_index('Method')['CV MSE (tscv)'] for t, r in all_cv.items()}, axis=1)

print("\n" + "="*70)
print("CROSS-ASSET SUMMARY — CV MSE (TimeSeriesSplit, tuning ≤2015)")
print("="*70)
print(summary_cv.to_string(float_format='%.6f'))

print("\n" + "="*70)
print("CROSS-ASSET SUMMARY — Validation MSE (2016-2020, expanding window)")
print("="*70)
print(summary_val_mse.to_string(float_format='%.6f'))

print("\n" + "="*70)
print("CROSS-ASSET SUMMARY — Validation OOS R² (2016-2020, expanding window)")
print("="*70)
print(summary_val_r2.to_string(float_format='%.6f'))

print("\n" + "="*70)
print("CROSS-ASSET SUMMARY — Validation IC (Spearman, 2016-2020)")
print("="*70)
print(summary_val_ic.to_string(float_format='%.6f'))

print("\n" + "="*70)
print("CROSS-ASSET SUMMARY — Test MSE (2021+)")
print("="*70)
print(summary_oos_mse.to_string(float_format='%.6f'))

print("\n" + "="*70)
print("CROSS-ASSET SUMMARY — Test OOS R² (Campbell-Thompson, 2021+)")
print("="*70)
print(summary_oos_r2.to_string(float_format='%.6f'))

print("\n" + "="*70)
print("CROSS-ASSET SUMMARY — Test IC (Spearman, 2021+)")
print("="*70)
print(summary_oos_ic.to_string(float_format='%.6f'))

summary_cv.to_csv(os.path.join(_OUT_DIR, 'model_selection_results_cv.csv'))
summary_val_mse.to_csv(os.path.join(_OUT_DIR, 'model_selection_results_val_mse.csv'))
summary_val_r2.to_csv(os.path.join(_OUT_DIR, 'model_selection_results_val_r2.csv'))
summary_val_ic.to_csv(os.path.join(_OUT_DIR, 'model_selection_results_val_ic.csv'))
summary_oos_mse.to_csv(os.path.join(_OUT_DIR, 'model_selection_results_oos.csv'))
summary_oos_r2.to_csv(os.path.join(_OUT_DIR, 'model_selection_results_oos_r2.csv'))
summary_oos_ic.to_csv(os.path.join(_OUT_DIR, 'model_selection_results_oos_ic.csv'))
print(f"\nSaved: {_OUT_DIR}/model_selection_results_{{cv, val_mse, val_r2, val_ic, oos, oos_r2, oos_ic}}.csv")

# ══════════════════════════════════════════════════════════════════════════════
#  ADDITIONAL OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

# In-sample stats for all targets → single CSV
pd.concat(
    [df.assign(target=t) for t, df in all_insam.items()]
).to_csv(os.path.join(_OUT_DIR, 'insample_stats_all.csv'), index=False)
print(f"Saved: {_OUT_DIR}/insample_stats_all.csv")

# Selected variables + coefficients for all targets → single CSV
if all_coefs:
    pd.concat(
        [df.assign(target=t) for t, df in all_coefs.items()]
    ).to_csv(os.path.join(_OUT_DIR, 'selected_vars_coefs.csv'), index=False)
    print(f"Saved: {_OUT_DIR}/selected_vars_coefs.csv")

# y_pred: one column per (model, target), separated into val and test CSVs
_pred_val, _pred_test = {}, {}
for _tgt, _method_dict in all_preds.items():
    for _key, _series in _method_dict.items():
        _phase, _mname = _key.split('_', 1)
        _col = f'{_mname}_{_tgt}'
        if _phase == 'val':
            _pred_val[_col] = _series
        else:
            _pred_test[_col] = _series
pd.DataFrame(_pred_val).to_csv(os.path.join(_OUT_DIR, 'y_pred_val.csv'))
pd.DataFrame(_pred_test).to_csv(os.path.join(_OUT_DIR, 'y_pred_test.csv'))
print(f"Saved: {_OUT_DIR}/y_pred_val.csv")
print(f"Saved: {_OUT_DIR}/y_pred_test.csv")
