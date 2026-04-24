import numpy as np
import pandas as pd
from functools import partial
from matplotlib.pyplot import subplots, close
from statsmodels.api import OLS, add_constant
import sklearn.model_selection as skm
import sklearn.linear_model as skl
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.base import BaseEstimator, TransformerMixin
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

# ══════════════════════════════════════════════════════════════════════════════
#  CUSTOM TRANSFORMER
# ══════════════════════════════════════════════════════════════════════════════

class StdWithInteractions(BaseEstimator, TransformerMixin):
    """Standardise base features then append surp×VIX interaction columns.
    fit() learns mean/std from train only → per-fold correct when used inside Pipeline."""
    def __init__(self, surp_idx, vix_idx):
        self.surp_idx = surp_idx
        self.vix_idx  = vix_idx

    def fit(self, X, y=None):
        self.scaler_ = StandardScaler().fit(X)
        return self

    def transform(self, X):
        Xs   = self.scaler_.transform(X)
        ints = Xs[:, self.surp_idx] * Xs[:, [self.vix_idx]]
        return np.hstack([Xs, ints])

# ══════════════════════════════════════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════════════════════════════════════

df = pd.read_csv("merged_announcement_panel.csv", index_col=0, parse_dates=True, dayfirst=True)
df = df.sort_index()

tune_df    = df.loc[:'2015-12-31']
val_df     = df.loc['2016-01-01':'2020-12-31']
test_df    = df.loc['2021-01-01':]
pretest_df = df.loc[:'2020-12-31']

ret_cols  = [c for c in df.columns if c.startswith('ret_') and '_lag' not in c]
surp_cols = [c for c in df.columns if c.startswith('surp_')]

K    = 3
tscv = TimeSeriesSplit(n_splits=K)

print(f"Tune:  {tune_df.index.min().date()} → {tune_df.index.max().date()}  ({len(tune_df)} obs)")
print(f"Val:   {val_df.index.min().date()} → {val_df.index.max().date()}  ({len(val_df)} obs)")
print(f"Test:  {test_df.index.min().date()} → {test_df.index.max().date()}  ({len(test_df)} obs)")
print(f"Targets: {ret_cols}")
print(f"Predictors per target: {len(surp_cols)} surp + 1 VIX [+ AR1 if daily] + {len(surp_cols)} VIX×surp interactions")

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

def _cv_mse_of_vars(vars_list, X_raw, predictor_names, Y, cv):
    """CV MSE with per-fold StandardScaler + LinearRegression on the given variable subset.
    X_raw: feature matrix (may be pre-scaled; scaler is a near-no-op if already standardised)."""
    if not vars_list:
        pipe = Pipeline([('lr', skl.LinearRegression())])
        X_cv = np.zeros((len(Y), 1))
    else:
        col_idx = [predictor_names.index(v) for v in vars_list]
        X_cv    = X_raw[:, col_idx]
        pipe    = Pipeline([('sc', StandardScaler()), ('lr', skl.LinearRegression())])
    return float(-skm.cross_validate(
        pipe, X_cv, Y, cv=cv, scoring='neg_mean_squared_error')['test_score'].mean())

# ── Module-level helpers for parallelised stepwise CV fold loops ──────────────
# Must be at module level (not closures) so joblib's loky backend can pickle them.

def _fold_dfs_fn(tr_idx, te_idx, X_tune, X_tune_df, base_predictors,
                 int_names, surp_idx, vix_idx):
    """Per-fold: fit StandardScaler on train split, transform both splits,
    then append surp×VIX interaction columns (computed from per-fold scaled values)."""
    all_pred = base_predictors + int_names
    fs = StandardScaler().fit(X_tune[tr_idx])

    def _with_ints(X_s):
        ints = X_s[:, surp_idx] * X_s[:, [vix_idx]]
        return np.hstack([X_s, ints])

    Xtr = pd.DataFrame(_with_ints(fs.transform(X_tune[tr_idx])),
                        columns=all_pred, index=X_tune_df.index[tr_idx])
    Xte = pd.DataFrame(_with_ints(fs.transform(X_tune[te_idx])),
                        columns=all_pred, index=X_tune_df.index[te_idx])
    return Xtr, Xte

def _fwd_fold_worker(tr_idx, te_idx, X_tune, X_tune_df, base_predictors,
                     int_names, surp_idx, vix_idx,
                     design, Y_tune, n_steps_cv, max_steps):
    """Build the full forward selection path on one CV fold and return per-step MSE."""
    Xf_tr, Xf_te = _fold_dfs_fn(tr_idx, te_idx, X_tune, X_tune_df, base_predictors,
                                  int_names, surp_idx, vix_idx)
    _path = sklearn_selection_path(OLS, Stepwise.fixed_steps(
        design, max_steps, direction='forward'))
    _path.fit(Xf_tr, Y_tune[tr_idx])
    _preds = _path.predict(Xf_te)
    n_s = min(_preds.shape[1], n_steps_cv)
    fe  = np.full((n_steps_cv,), np.nan)
    fe[:n_s] = ((_preds[:, :n_s] - Y_tune[te_idx, None]) ** 2).mean(0)
    return fe

def _bwd_fold_worker(tr_idx, te_idx, X_tune, X_tune_df, base_predictors,
                     int_names, surp_idx, vix_idx,
                     design, Y_tune, n_steps_bwd, min_terms):
    """Build the full backward selection path on one CV fold and return per-step MSE."""
    Xf_tr, Xf_te = _fold_dfs_fn(tr_idx, te_idx, X_tune, X_tune_df, base_predictors,
                                  int_names, surp_idx, vix_idx)
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
    """Expanding-window eval for builder-based models.
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

n_targets = len(ret_cols)
_fkw = dict(squeeze=False)
fig_fwd,   axes_fwd   = subplots(1, n_targets, figsize=(8 * n_targets, 5), **_fkw)
fig_bwd,   axes_bwd   = subplots(1, n_targets, figsize=(8 * n_targets, 5), **_fkw)
fig_rpath, axes_rpath = subplots(1, n_targets, figsize=(8 * n_targets, 5), **_fkw)
fig_rcv,   axes_rcv   = subplots(1, n_targets, figsize=(7 * n_targets, 5), **_fkw)
fig_lpath, axes_lpath = subplots(1, n_targets, figsize=(8 * n_targets, 5), **_fkw)
fig_lcv,   axes_lcv   = subplots(1, n_targets, figsize=(7 * n_targets, 5), **_fkw)
fig_dim,   axes_dim   = subplots(1, n_targets, figsize=(8 * n_targets, 5), **_fkw)
fig_oos,   axes_oos   = subplots(1, n_targets, figsize=(10 * n_targets, 5), **_fkw)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP — one full model comparison per return target
# ══════════════════════════════════════════════════════════════════════════════

all_oos   = {}
all_cv    = {}
all_insam = {}
all_val   = {}
all_preds = {}   # {target: {phase_method: pd.Series(preds, index=dates)}}
all_coefs = {}   # {target: pd.DataFrame of selected vars + coefficients}

for ti, target in enumerate(ret_cols):

    print(f"\n{'='*70}")
    print(f"  TARGET: {target}")
    print(f"{'='*70}")

    ar_term         = f"{target}_lag1"
    has_ar1         = ar_term in df.columns
    base_predictors = ([ar_term] if has_ar1 else []) + surp_cols + ['VIXCLS']
    int_names       = [f"{s}_x_vix" for s in surp_cols]
    all_predictors  = base_predictors + int_names

    # Indices for StdWithInteractions and per-fold interaction computation
    _surp_idx = [base_predictors.index(s) for s in surp_cols]
    _vix_idx  = base_predictors.index('VIXCLS')

    # ── Clean rows — separate tune / val / pretest / test ─────────────────────
    full_clean    = df[[target] + base_predictors].dropna()
    tune_clean    = full_clean.loc[:'2015-12-31']
    val_clean     = full_clean.loc['2016-01-01':'2020-12-31']
    pretest_clean = full_clean.loc[:'2020-12-31']
    test_clean    = full_clean.loc['2021-01-01':]

    Y_tune       = tune_clean[target].values
    X_tune_raw   = tune_clean[base_predictors].values     # raw, unstandardised
    X_tune_df    = tune_clean[base_predictors]            # DataFrame for ISLP
    n_tune       = len(Y_tune)

    Y_val     = val_clean[target].values
    Y_pretest = pretest_clean[target].values
    Y_test    = test_clean[target].values

    print(f"  Tune obs: {n_tune}  |  Val obs: {len(Y_val)}  |  Test obs: {len(Y_test)}")
    print(f"  has_ar1={has_ar1}  base={len(base_predictors)}  total (with ints)={len(all_predictors)}")

    # ── Global scaler for ISLP only (mild leakage; used for spec selection) ────
    _scaler_islp = StandardScaler().fit(X_tune_raw)
    X_tune_s     = _scaler_islp.transform(X_tune_raw)

    # Build globally-scaled DataFrame with interaction columns for ISLP design
    int_tune        = X_tune_s[:, _surp_idx] * X_tune_s[:, [_vix_idx]]
    X_tune_s_int    = np.hstack([X_tune_s, int_tune])
    X_tune_s_int_df = pd.DataFrame(X_tune_s_int, columns=all_predictors, index=tune_clean.index)

    # Also fit StdWithInteractions once for Ridge/Lasso closed-form path plots
    _swi_path  = StdWithInteractions(_surp_idx, _vix_idx).fit(X_tune_raw)
    X_tune_swi = _swi_path.transform(X_tune_raw)   # n_tune × len(all_predictors)

    # ── AR(1) benchmark ───────────────────────────────────────────────────────
    if has_ar1:
        ar_idx = base_predictors.index(ar_term)
        ar1_pipe = Pipeline([('sc', StandardScaler()), ('lr', skl.LinearRegression())])
        ar1_cv_mse = float(-skm.cross_validate(
            ar1_pipe, X_tune_raw[:, [ar_idx]], Y_tune,
            cv=tscv, scoring='neg_mean_squared_error')['test_score'].mean())
        ar1_pipe.fit(X_tune_raw[:, [ar_idx]], Y_tune)
    else:
        ar_idx     = None
        ar1_cv_mse = np.nan

    # ── sigma2 and ISLP design (tuning data, globally scaled + interactions) ──
    design           = MS(all_predictors).fit(X_tune_s_int_df)
    X_with_intercept = design.transform(X_tune_s_int_df)
    sigma2           = OLS(Y_tune, X_with_intercept).fit().scale
    neg_Cp           = partial(nCp, sigma2)

    D_tune = design.fit_transform(X_tune_s_int_df).drop('intercept', axis=1)

    # ── Per-fold closure (delegates to module-level _fold_dfs_fn) ─────────────
    def _fold_dfs(tr_idx, te_idx):
        return _fold_dfs_fn(tr_idx, te_idx, X_tune_raw, X_tune_df,
                            base_predictors, int_names, _surp_idx, _vix_idx)

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE 1: SUBSET SELECTION — spec selection via CV on tuning data
    # ══════════════════════════════════════════════════════════════════════════

    # -- Forward Cp -----------------------------------------------------------
    fwd_strategy_cp = Stepwise.first_peak(design, direction='forward',
                                           max_terms=len(design.terms))
    forward_Cp = sklearn_selected(OLS, fwd_strategy_cp, scoring=neg_Cp)
    forward_Cp.fit(X_tune_s_int_df, Y_tune)
    fwd_cp_vars   = [c for c in D_tune.columns if c in forward_Cp.selected_state_]
    fwd_cp_cv_mse = _cv_mse_of_vars(fwd_cp_vars, X_tune_swi, all_predictors, Y_tune, tscv)

    # -- Forward CV path ------------------------------------------------------
    fwd_strategy_cv = Stepwise.fixed_steps(design, len(design.terms), direction='forward')
    forward_cv_path = sklearn_selection_path(OLS, fwd_strategy_cv)
    forward_cv_path.fit(X_tune_s_int_df, Y_tune)
    Yhat_fwd_in     = forward_cv_path.predict(X_tune_s_int_df)   # (n, n_steps)
    n_steps_cv      = len(forward_cv_path.models_)

    cv_fwd_mse = np.array(
        Parallel(n_jobs=-1, verbose=10)(
            delayed(_fwd_fold_worker)(
                tr_idx, te_idx, X_tune_raw, X_tune_df, base_predictors,
                int_names, _surp_idx, _vix_idx,
                design, Y_tune, n_steps_cv, len(design.terms)
            )
            for tr_idx, te_idx in tscv.split(Y_tune)
        )
    ).T   # (n_steps, K)

    best_cv_step  = int(np.nanargmin(np.nanmean(cv_fwd_mse, axis=1)))
    best_cv_state = forward_cv_path.models_[best_cv_step][0]
    fwd_cv_vars   = _state_to_D_columns(best_cv_state, D_tune.columns)

    # Validation curve for plot (last 20% of tuning data — for visualisation only)
    _plot_val_cut = int(n_tune * 0.8)
    forward_cv_path.fit(X_tune_s_int_df.iloc[:_plot_val_cut], Y_tune[:_plot_val_cut])
    Yhat_val_fwd = forward_cv_path.predict(X_tune_s_int_df.iloc[_plot_val_cut:])
    val_mse_fwd  = ((Yhat_val_fwd - Y_tune[_plot_val_cut:, None]) ** 2).mean(0)
    forward_cv_path.fit(X_tune_s_int_df, Y_tune)   # refit on full tuning data

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
    backward_Cp.fit(X_tune_s_int_df, Y_tune)
    bwd_cp_vars   = [c for c in D_tune.columns if c in backward_Cp.selected_state_]
    bwd_cp_cv_mse = _cv_mse_of_vars(bwd_cp_vars, X_tune_swi, all_predictors, Y_tune, tscv)

    # -- Backward CV path -----------------------------------------------------
    bwd_strategy_cv  = Stepwise.fixed_steps(design, 0, direction='backward',
                                             initial_terms=list(design.terms))
    backward_cv_path = sklearn_selection_path(OLS, bwd_strategy_cv)
    backward_cv_path.fit(X_tune_s_int_df, Y_tune)
    Yhat_bwd_in      = backward_cv_path.predict(X_tune_s_int_df)
    n_steps_bwd      = len(backward_cv_path.models_)

    cv_bwd_mse = np.array(
        Parallel(n_jobs=-1, verbose=10)(
            delayed(_bwd_fold_worker)(
                tr_idx, te_idx, X_tune_raw, X_tune_df, base_predictors,
                int_names, _surp_idx, _vix_idx,
                design, Y_tune, n_steps_bwd, 0
            )
            for tr_idx, te_idx in tscv.split(Y_tune)
        )
    ).T   # (n_steps, K)

    best_bwd_step  = int(np.nanargmin(np.nanmean(cv_bwd_mse, axis=1)))
    best_bwd_state = backward_cv_path.models_[best_bwd_step][0]
    bwd_cv_vars    = _state_to_D_columns(best_bwd_state, D_tune.columns)

    # Validation curve for backward (last 20% of tuning data — for visualisation only)
    backward_cv_path.fit(X_tune_s_int_df.iloc[:_plot_val_cut], Y_tune[:_plot_val_cut])
    Yhat_val_bwd = backward_cv_path.predict(X_tune_s_int_df.iloc[_plot_val_cut:])
    val_mse_bwd  = ((Yhat_val_bwd - Y_tune[_plot_val_cut:, None]) ** 2).mean(0)
    backward_cv_path.fit(X_tune_s_int_df, Y_tune)   # refit on full tuning data

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
    Yhat_fwd_cv_t = forward_cv_path.predict(X_tune_s_int_df)[:, best_cv_step]
    Yhat_bwd_cp   = _lr_yhat(bwd_cp_vars, D_tune)
    Yhat_bwd_cv_t = backward_cv_path.predict(X_tune_s_int_df)[:, best_bwd_step]

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
    #  Pipeline([StdWithInteractions, model]) → per-fold scaling + interactions
    # ══════════════════════════════════════════════════════════════════════════

    # RIDGE
    lambdas    = 10 ** np.linspace(8, -6, 100) / Y_tune.std()
    grid_ridge = skm.GridSearchCV(
        Pipeline([('swi', StdWithInteractions(_surp_idx, _vix_idx)), ('ridge', skl.Ridge())]),
        {'ridge__alpha': lambdas}, cv=tscv, scoring='neg_mean_squared_error',
        n_jobs=-1)
    grid_ridge.fit(X_tune_raw, Y_tune)
    ridge_alpha  = grid_ridge.best_params_['ridge__alpha']
    ridge_cv_mse = -grid_ridge.best_score_

    # Ridge solution-path plot — closed-form on globally-scaled+int data
    _XtX = X_tune_swi.T @ X_tune_swi
    _XtY = X_tune_swi.T @ Y_tune
    _eye = np.eye(X_tune_swi.shape[1])
    soln_ridge = np.array([
        np.linalg.solve(_XtX + a * _eye, _XtY)
        for a in lambdas
    ])

    soln_ridge_df = pd.DataFrame(soln_ridge, columns=all_predictors, index=np.log(lambdas))
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
    # LASSO
    lasso_alphas = np.logspace(-6, 2, 100)
    grid_lasso   = skm.GridSearchCV(
        Pipeline([('swi', StdWithInteractions(_surp_idx, _vix_idx)),
                  ('lasso', skl.Lasso(max_iter=10000))]),
        {'lasso__alpha': lasso_alphas}, cv=tscv, scoring='neg_mean_squared_error',
        n_jobs=-1)
    grid_lasso.fit(X_tune_raw, Y_tune)
    lasso_alpha  = grid_lasso.best_params_['lasso__alpha']
    lasso_cv_mse = -grid_lasso.best_score_
    p_lasso_nz   = int((np.abs(grid_lasso.best_estimator_['lasso'].coef_) > 1e-8).sum())

    print("Lasso - Done")
    # Lasso solution path on globally-scaled+int data
    alphas_path, soln_lasso, _ = skl.Lasso.path(X_tune_swi, Y_tune, alphas=lasso_alphas, l1_ratio=1.0)
    if alphas_path[0] > alphas_path[-1]:
        alphas_path = alphas_path[::-1]
        soln_lasso  = soln_lasso[:, ::-1]
    soln_lasso_df = pd.DataFrame(soln_lasso.T, columns=all_predictors,
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

    # Ridge vs Lasso coefficient table (indexed to all_predictors)
    coef_table = pd.DataFrame({
        'Ridge': grid_ridge.best_estimator_['ridge'].coef_,
        'Lasso': grid_lasso.best_estimator_['lasso'].coef_,
    }, index=all_predictors)
    coef_table['Lasso selected'] = coef_table['Lasso'].abs() > 1e-8
    print(f"\n--- Ridge vs Lasso coefficients ({target}) ---")
    print(coef_table.to_string(float_format='%.5f'))

    # PCR
    max_comp = min(50, len(all_predictors), X_tune_raw.shape[0] - 1)
    grid_pcr = skm.GridSearchCV(
        Pipeline([('swi', StdWithInteractions(_surp_idx, _vix_idx)),
                  ('pca', PCA()), ('lr', skl.LinearRegression())]),
        {'pca__n_components': range(1, max_comp + 1)},
        cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_pcr.fit(X_tune_raw, Y_tune)
    pcr_best_n = grid_pcr.best_params_['pca__n_components']
    pcr_cv_mse = -grid_pcr.best_score_

    print("PCR - Done")
    # PLS
    max_comp_pls = min(20, len(all_predictors))
    grid_pls = skm.GridSearchCV(
        Pipeline([('swi', StdWithInteractions(_surp_idx, _vix_idx)),
                  ('pls', PLSRegression(scale=False))]),
        {'pls__n_components': range(1, max_comp_pls + 1)},
        cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_pls.fit(X_tune_raw, Y_tune)
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

    Yhat_ridge = grid_ridge.best_estimator_.predict(X_tune_raw)
    Yhat_lasso = grid_lasso.best_estimator_.predict(X_tune_raw)
    Yhat_pcr   = grid_pcr.best_estimator_.predict(X_tune_raw)
    Yhat_pls   = grid_pls.best_estimator_.predict(X_tune_raw).ravel()

    # Table 1: CV MSE (tuning set)
    cv_rows = []
    if has_ar1:
        cv_rows.append(('AR(1) benchmark', ar1_cv_mse, '1 predictor'))
    cv_rows += [
        ('Fwd (Cp)',  fwd_cp_cv_mse, f'{len(fwd_cp_vars)} pred.'),
        ('Fwd (CV)',  float(np.nanmean(cv_fwd_mse, axis=1)[best_cv_step]),
                      f'{best_cv_step} pred.'),
        ('Bwd (Cp)',  bwd_cp_cv_mse, f'{len(bwd_cp_vars)} pred.'),
        ('Bwd (CV)',  float(np.nanmean(cv_bwd_mse, axis=1)[best_bwd_step]),
                      f'{best_bwd_step} removed'),
        ('Ridge',     ridge_cv_mse,  f'alpha={ridge_alpha:.2e}'),
        ('Lasso',     lasso_cv_mse,  f'alpha={lasso_alpha:.2e}, nz={p_lasso_nz}'),
        ('PCR',       pcr_cv_mse,    f'{pcr_best_n} comp.'),
        ('PLS',       pls_cv_mse,    f'{pls_best_n} comp.'),
    ]
    cv_df = pd.DataFrame(cv_rows, columns=['Method', 'CV MSE (tscv)', 'Best param'])
    all_cv[target] = cv_df
    print(f"\n--- Table 1: CV MSE ({target}, tuning ≤2015) ---")
    print(cv_df.to_string(index=False, float_format='%.6f'))

    # Table 2: in-sample stats (tuning set)
    insam_entries = []
    if has_ar1:
        Yhat_ar1 = ar1_pipe.predict(X_tune_raw[:, [ar_idx]])
        insam_entries.append(('AR(1) benchmark', Yhat_ar1, 1))
    insam_entries += [
        ('Fwd (Cp)',  Yhat_fwd_cp,   len(fwd_cp_vars)),
        ('Fwd (CV)',  Yhat_fwd_cv_t, len(fwd_cv_vars)),
        ('Bwd (Cp)',  Yhat_bwd_cp,   len(bwd_cp_vars)),
        ('Bwd (CV)',  Yhat_bwd_cv_t, len(bwd_cv_vars)),
        ('Ridge',     Yhat_ridge,    len(all_predictors)),
        ('Lasso',     Yhat_lasso,    p_lasso_nz),
        ('PCR',       Yhat_pcr,      pcr_best_n),
        ('PLS',       Yhat_pls,      pls_best_n),
    ]
    insam_rows = []
    for name, yh, p_model in insam_entries:
        adj_r2, cp_val, bic = insample_stats(Y_tune, yh, p_model, sigma2)
        insam_rows.append({'Method': name, 'p': p_model,
                           'Adj.R2': adj_r2, 'Cp': cp_val, 'BIC': bic})
    insam_df = pd.DataFrame(insam_rows)
    all_insam[target] = insam_df
    print(f"\n--- Table 2: In-sample stats ({target}, tuning ≤2015) ---")
    print(insam_df.to_string(index=False, float_format='%.5f'))

    # ══════════════════════════════════════════════════════════════════════════
    #  SELECTED VARIABLES + COEFFICIENTS
    #  Subset methods: NW HAC SEs; Ridge/Lasso: coefs as-is
    # ══════════════════════════════════════════════════════════════════════════

    nw_lags = max(1, int(4 * (n_tune / 100) ** (2 / 9)))
    coef_rows = []
    for mname_c, vars_list_c in [
        ('Fwd (Cp)', fwd_cp_vars),
        ('Fwd (CV)', fwd_cv_vars),
        ('Bwd (Cp)', bwd_cp_vars),
        ('Bwd (CV)', bwd_cv_vars),
    ]:
        if vars_list_c:
            idx_c  = [all_predictors.index(v) for v in vars_list_c]
            _sc_c  = StandardScaler().fit(X_tune_swi[:, idx_c])
            X_sub  = add_constant(_sc_c.transform(X_tune_swi[:, idx_c]))
            res    = OLS(Y_tune, X_sub).fit(
                cov_type='HAC',
                cov_kwds={'maxlags': nw_lags, 'use_correction': True})
            for v, coef, se, tstat, pval in zip(
                    vars_list_c, res.params[1:], res.bse[1:],
                    res.tvalues[1:], res.pvalues[1:]):
                coef_rows.append({'Method': mname_c, 'Variable': v,
                                  'Coefficient': coef, 'SE': se,
                                  't-stat': tstat, 'p-value': pval})
    for v, c in zip(all_predictors, grid_ridge.best_estimator_['ridge'].coef_):
        coef_rows.append({'Method': 'Ridge', 'Variable': v, 'Coefficient': c,
                          'SE': np.nan, 't-stat': np.nan, 'p-value': np.nan})
    for v, c in zip(all_predictors, grid_lasso.best_estimator_['lasso'].coef_):
        if abs(c) > 1e-8:
            coef_rows.append({'Method': 'Lasso', 'Variable': v, 'Coefficient': c,
                              'SE': np.nan, 't-stat': np.nan, 'p-value': np.nan})
    all_coefs[target] = pd.DataFrame(coef_rows)

    # ══════════════════════════════════════════════════════════════════════════
    #  RANDOM FOREST + LIGHTGBM — raw base features (no scaling needed for trees)
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
        resource='n_samples',   # explicit: adds rows front-to-back → chronological
    )

    rf_search = HalvingRandomSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        rf_param_dist, n_candidates=60, **_halving_kw,
    )
    rf_search.fit(X_tune_raw, Y_tune)
    rf_best   = rf_search.best_params_
    rf_cv_mse = -rf_search.best_score_
    print(f"\n  [RF]  Best tune CV-MSE: {rf_cv_mse:.6f}")
    print(f"  [RF]  Best params:      {rf_best}")

    lgb_search = HalvingRandomSearchCV(
        LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
        lgb_param_dist, n_candidates=60, **_halving_kw,
    )
    lgb_search.fit(X_tune_raw, Y_tune)
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
    rf_tune_model.fit(X_tune_raw, Y_tune)
    lgb_tune_model = LGBMRegressor(**lgb_best, random_state=42, n_jobs=-1, verbose=-1)
    lgb_tune_model.fit(X_tune_raw, Y_tune)

    # Feature importance plot (top 20 per model)
    _top_n = 20
    fig_fi, axes_fi = plt.subplots(1, 2, figsize=(16, 6))
    for ax_fi, fi_model, fi_label in [
        (axes_fi[0], rf_tune_model,  'RF'),
        (axes_fi[1], lgb_tune_model, 'LGB'),
    ]:
        imp = pd.Series(fi_model.feature_importances_, index=base_predictors).nlargest(_top_n)
        ax_fi.barh(imp.index[::-1], imp.values[::-1])
        ax_fi.set_xlabel('Importance', fontsize=10)
        ax_fi.set_title(f'{fi_label} Feature Importance — {target}', fontsize=11)
    fig_fi.tight_layout()
    fig_fi.savefig(f'outcomes/feature_importance_{target}.png', dpi=120)
    plt.close(fig_fi)

    # SHAP summary plots (beeswarm, on tuning data)
    for shap_model, shap_name in [(rf_tune_model, 'rf'), (lgb_tune_model, 'lgb')]:
        shap_explainer = shap.TreeExplainer(shap_model)
        shap_vals = shap_explainer.shap_values(X_tune_raw)
        fig_shap = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals, X_tune_raw, feature_names=base_predictors, show=False)
        plt.title(f'SHAP Summary — {shap_name.upper()} — {target}')
        plt.tight_layout()
        fig_shap.savefig(f'outcomes/shap_summary_{shap_name}_{target}.png', dpi=120,
                         bbox_inches='tight')
        plt.close(fig_shap)

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE 2: EXPANDING-WINDOW VALIDATION (2016-2020)
    #  Model specs (variable sets, hyperparameters) are fixed from Phase 1.
    #  Only coefficients are refit at each step of the expanding window.
    # ══════════════════════════════════════════════════════════════════════════

    _swi_spec    = {'surp_idx': _surp_idx, 'vix_idx': _vix_idx}
    _ar1_spec    = {'has_ar1': has_ar1, 'ar_idx': ar_idx}
    _subset_spec = {'all_predictors': all_predictors, 'surp_idx': _surp_idx, 'vix_idx': _vix_idx}

    def _build_ar1(X_tr, Y_tr, spec):
        if not spec.get('has_ar1', False):
            mean_val = float(Y_tr.mean())
            return lambda X_te: mean_val
        ar_i = spec['ar_idx']
        pipe = Pipeline([('sc', StandardScaler()), ('lr', skl.LinearRegression())])
        pipe.fit(X_tr[:, [ar_i]], Y_tr)
        return lambda X_te: pipe.predict(X_te[:, [ar_i]])[0]

    def _build_subset(X_tr, Y_tr, spec):
        """X_tr is raw base features. Applies StdWithInteractions then selects vars."""
        vl = spec['vars_list']
        if not vl:
            mean_val = float(Y_tr.mean())
            return lambda X_te: mean_val
        swi = StdWithInteractions(spec['surp_idx'], spec['vix_idx'])
        swi.fit(X_tr)
        ap  = spec['all_predictors']
        idx = [ap.index(v) for v in vl]
        lr  = skl.LinearRegression().fit(swi.transform(X_tr)[:, idx], Y_tr)
        return lambda X_te: lr.predict(swi.transform(X_te)[:, idx])[0]

    def _build_ridge(X_tr, Y_tr, spec):
        pipe = Pipeline([('swi', StdWithInteractions(spec['surp_idx'], spec['vix_idx'])),
                         ('ridge', skl.Ridge(alpha=spec['alpha']))])
        pipe.fit(X_tr, Y_tr)
        return lambda X_te: pipe.predict(X_te)[0]

    def _build_lasso(X_tr, Y_tr, spec):
        pipe = Pipeline([('swi', StdWithInteractions(spec['surp_idx'], spec['vix_idx'])),
                         ('lasso', skl.Lasso(alpha=spec['alpha'], max_iter=10000))])
        pipe.fit(X_tr, Y_tr)
        return lambda X_te: pipe.predict(X_te)[0]

    def _build_pcr(X_tr, Y_tr, spec):
        pipe = Pipeline([('swi', StdWithInteractions(spec['surp_idx'], spec['vix_idx'])),
                         ('pca', PCA(n_components=spec['n_comp'])),
                         ('lr', skl.LinearRegression())])
        pipe.fit(X_tr, Y_tr)
        return lambda X_te: pipe.predict(X_te)[0]

    def _build_pls(X_tr, Y_tr, spec):
        pipe = Pipeline([('swi', StdWithInteractions(spec['surp_idx'], spec['vix_idx'])),
                         ('pls', PLSRegression(n_components=spec['n_comp'], scale=False))])
        pipe.fit(X_tr, Y_tr)
        return lambda X_te: pipe.predict(X_te)[0]

    def _build_rf(X_tr, Y_tr, spec):
        m = RandomForestRegressor(**spec['params'], random_state=42, n_jobs=-1)
        m.fit(X_tr, Y_tr)
        return lambda X_te: m.predict(X_te)[0]

    def _build_lgb(X_tr, Y_tr, spec):
        m = LGBMRegressor(**spec['params'], random_state=42, n_jobs=-1, verbose=-1)
        m.fit(X_tr, Y_tr)
        return lambda X_te: m.predict(X_te)[0]

    val_window_df = pretest_clean   # raw base features up to 2020-12-31

    methods_val = []
    if has_ar1:
        methods_val.append(('AR(1)', _build_ar1, {**_ar1_spec}))
    methods_val += [
        ('Fwd (Cp)',  _build_subset, {**_subset_spec, 'vars_list': fwd_cp_vars}),
        ('Fwd (CV)',  _build_subset, {**_subset_spec, 'vars_list': fwd_cv_vars}),
        ('Bwd (Cp)',  _build_subset, {**_subset_spec, 'vars_list': bwd_cp_vars}),
        ('Bwd (CV)',  _build_subset, {**_subset_spec, 'vars_list': bwd_cv_vars}),
        ('Ridge',     _build_ridge,  {**_swi_spec, 'alpha': ridge_alpha}),
        ('Lasso',     _build_lasso,  {**_swi_spec, 'alpha': lasso_alpha}),
        ('PCR',       _build_pcr,    {**_swi_spec, 'n_comp': pcr_best_n}),
        ('PLS',       _build_pls,    {**_swi_spec, 'n_comp': pls_best_n}),
        ('RF',        _build_rf,     {'params': rf_best}),
        ('LGB',       _build_lgb,    {'params': lgb_best}),
    ]

    val_results = []
    for mname, builder, spec in methods_val:
        y_act, y_pred, pred_dates = periodic_expanding_window_eval_builder(
            builder, val_clean.index, val_window_df, target, base_predictors, spec,
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

    test_window_df = full_clean   # raw base features, full history

    methods_test = [
        ('AR(1) benchmark', _build_ar1,    {**_ar1_spec}),
        ('Fwd (Cp)',        _build_subset, {**_subset_spec, 'vars_list': fwd_cp_vars}),
        ('Fwd (CV)',        _build_subset, {**_subset_spec, 'vars_list': fwd_cv_vars}),
        ('Bwd (Cp)',        _build_subset, {**_subset_spec, 'vars_list': bwd_cp_vars}),
        ('Bwd (CV)',        _build_subset, {**_subset_spec, 'vars_list': bwd_cv_vars}),
        ('Ridge',           _build_ridge,  {**_swi_spec, 'alpha': ridge_alpha}),
        ('Lasso',           _build_lasso,  {**_swi_spec, 'alpha': lasso_alpha}),
        ('PCR',             _build_pcr,    {**_swi_spec, 'n_comp': pcr_best_n}),
        ('PLS',             _build_pls,    {**_swi_spec, 'n_comp': pls_best_n}),
        ('RF',              _build_rf,     {'params': rf_best}),
        ('LGB',             _build_lgb,    {'params': lgb_best}),
    ]
    if not has_ar1:
        methods_test = [m for m in methods_test if m[0] != 'AR(1) benchmark']

    oos_rows = []
    _all_test_preds = []
    for mname, builder, spec in methods_test:
        y_act, y_pred, pred_dates = periodic_expanding_window_eval_builder(
            builder, test_clean.index, test_window_df, target, base_predictors, spec,
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
    fig_cum.savefig(f'outcomes/cumulative_oos_{target}.png', dpi=120)
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
    (fig_fwd,   'outcomes/fwd_stepwise_summary.png'),
    (fig_bwd,   'outcomes/bwd_stepwise_summary.png'),
    (fig_rpath, 'outcomes/ridge_path_summary.png'),
    (fig_rcv,   'outcomes/ridge_cv_summary.png'),
    (fig_lpath, 'outcomes/lasso_path_summary.png'),
    (fig_lcv,   'outcomes/lasso_cv_summary.png'),
    (fig_dim,   'outcomes/pcr_pls_summary.png'),
    (fig_oos,   'outcomes/oos_r2_summary.png'),
]
for fig, fname in _summary_plots:
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    close(fig)
print("\nSaved:", [f for _, f in _summary_plots])

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

summary_cv.to_csv('outcomes/model_selection_results_cv.csv')
summary_val_mse.to_csv('outcomes/model_selection_results_val_mse.csv')
summary_val_r2.to_csv('outcomes/model_selection_results_val_r2.csv')
summary_val_ic.to_csv('outcomes/model_selection_results_val_ic.csv')
summary_oos_mse.to_csv('outcomes/model_selection_results_oos.csv')
summary_oos_r2.to_csv('outcomes/model_selection_results_oos_r2.csv')
summary_oos_ic.to_csv('outcomes/model_selection_results_oos_ic.csv')
print("\nSaved: outcomes/model_selection_results_{cv, val_mse, val_r2, val_ic, oos, oos_r2, oos_ic}.csv")

# ══════════════════════════════════════════════════════════════════════════════
#  ADDITIONAL OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

# In-sample stats for all targets → single CSV
pd.concat(
    [df.assign(target=t) for t, df in all_insam.items()]
).to_csv('outcomes/insample_stats_all.csv', index=False)
print("Saved: outcomes/insample_stats_all.csv")

# Selected variables + coefficients for all targets → single CSV
if all_coefs:
    pd.concat(
        [df.assign(target=t) for t, df in all_coefs.items()]
    ).to_csv('outcomes/selected_vars_coefs.csv', index=False)
    print("Saved: outcomes/selected_vars_coefs.csv")

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
pd.DataFrame(_pred_val).to_csv('outcomes/y_pred_val.csv')
pd.DataFrame(_pred_test).to_csv('outcomes/y_pred_test.csv')
print("Saved: outcomes/y_pred_val.csv")
print("Saved: outcomes/y_pred_test.csv")
