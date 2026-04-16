import os
import tempfile
import numpy as np
import pandas as pd
from functools import partial
from matplotlib.pyplot import subplots, close
from statsmodels.api import OLS
import sklearn.model_selection as skm
import sklearn.linear_model as skl
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import spearmanr
from ISLP.models import ModelSpec as MS
from ISLP.models import (Stepwise, sklearn_selected, sklearn_selection_path)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingRandomSearchCV
from lightgbm import LGBMRegressor


# Ensure Numba (used by l0bnb) has a writable cache dir
_numba_cache_dir = os.path.join(tempfile.gettempdir(), "numba_cache")
os.makedirs(_numba_cache_dir, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", _numba_cache_dir)
from l0bnb import fit_path

# ══════════════════════════════════════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════════════════════════════════════

df = pd.read_csv('test_set.csv', index_col=0, parse_dates=True, dayfirst=True)
df = df.sort_index()

tune_df    = df.loc[:'2015-12-31']                     # Step 1: CV tuning
val_df     = df.loc['2016-01-01':'2020-12-31']         # Step 2: expanding-window validation
test_df    = df.loc['2021-01-01':]                     # Step 3: final holdout
pretest_df = df.loc[:'2020-12-31']                     # for refitting before test evaluation

ret_cols  = [c for c in df.columns if c.startswith('ret_') and '_lag' not in c]
surp_cols = [c for c in df.columns if c.startswith('surp_')]

K    = 5
tscv = TimeSeriesSplit(n_splits=K)

print(f"Tune:  {tune_df.index.min().date()} → {tune_df.index.max().date()}  ({len(tune_df)} obs)")
print(f"Val:   {val_df.index.min().date()} → {val_df.index.max().date()}  ({len(val_df)} obs)")
print(f"Test:  {test_df.index.min().date()} → {test_df.index.max().date()}  ({len(test_df)} obs)")
print(f"Targets: {ret_cols}")
print(f"Predictors per target: 1 AR + {len(surp_cols)} surp + 1 VIX = {1+len(surp_cols)+1}")

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
    X_raw: raw (unscaled) feature matrix; standardization happens inside each fold."""
    if not vars_list:
        pipe = Pipeline([('lr', skl.LinearRegression())])
        X_cv = np.zeros((len(Y), 1))
    else:
        col_idx = [predictor_names.index(v) for v in vars_list]
        X_cv    = X_raw[:, col_idx]
        pipe    = Pipeline([('sc', StandardScaler()), ('lr', skl.LinearRegression())])
    return float(-skm.cross_validate(
        pipe, X_cv, Y, cv=cv, scoring='neg_mean_squared_error')['test_score'].mean())

def periodic_expanding_window_eval(model_cls, model_params, val_idx,
                                   window_df, target, predictors,
                                   refit_freq='MS'):
    """Expanding-window eval with periodic (default: monthly) refitting.
    Trains once at the start of each period on all data before that period,
    then predicts every day in the period with the same frozen model.
    'MS' = month start.  'QS' = quarter start if you want even faster."""
    y_preds, y_acts = [], []
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

    return np.array(y_acts), np.array(y_preds)


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

for ti, target in enumerate(ret_cols):

    print(f"\n{'='*70}")
    print(f"  TARGET: {target}")
    print(f"{'='*70}")

    ar_term    = f"{target}_lag1"
    predictors = [ar_term] + surp_cols + ['vix_lag1']

    # ── Clean rows — separate tune / val / pretest / test ─────────────────────
    full_clean    = df[[target] + predictors].dropna()
    tune_clean    = full_clean.loc[:'2015-12-31']
    val_clean     = full_clean.loc['2016-01-01':'2020-12-31']
    pretest_clean = full_clean.loc[:'2020-12-31']
    test_clean    = full_clean.loc['2021-01-01':]

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

    # ── Tuning-set standardization (for CV, in-sample stats, ISLP methods) ────
    scaler      = StandardScaler().fit(X_tune)
    X_tune_s    = scaler.transform(X_tune)
    X_tune_s_df = pd.DataFrame(X_tune_s, columns=predictors, index=X_tune_df.index)
    
    
        # ══════════════════════════════════════════════════════════════════════════
    #  RANDOM FOREST  +  LIGHTGBM
    # ══════════════════════════════════════════════════════════════════════════
    #
    # Speed strategy:
    #  (1) HalvingRandomSearchCV: eliminates weak candidates early using
    #      successively more data — ~5-8x faster than RandomizedSearchCV
    #      for the same number of initial candidates.
    #  (2) Periodic (monthly) expanding-window refit: ~1250 daily refits
    #      → ~60 monthly refits per target per model.
    #
    # Grids are intentionally tight for daily returns (low SNR, ~145 features):
    #  - RF: aggressive feature subsampling (sqrt ≈ 12 of 145), large leaf floors
    #  - LGB: small num_leaves, large min_child_samples, conservative lr
    # ─────────────────────────────────────────────────────────────────────────

    rf_param_dist = {
        'n_estimators':     [300, 500, 800],
        'max_depth':        [2, 3, 4],            # shallow — noise kills deep trees
        'max_features':     ['sqrt', 0.2, 0.3],   # sqrt≈12, 0.2≈29 of 145 features
        'min_samples_leaf': [15, 30, 50],          # large floor: don't split on noise
        'max_samples':      [0.6, 0.8],            # row-bagging fraction
    }

    lgb_param_dist = {
        'n_estimators':      [200, 300, 500],
        'learning_rate':     [0.01, 0.02, 0.05],
        'num_leaves':        [10, 20, 31],         # max tree complexity
        'min_child_samples': [30, 50, 100],        # most important guard vs noise
        'feature_fraction':  [0.2, 0.3, 0.5],     # 0.2 ≈ 29 of 145 features
        'bagging_fraction':  [0.7, 0.8],
        'bagging_freq':      [1],
        'reg_lambda':        [0.5, 1.0, 2.0],
    }

    # HalvingRandomSearchCV: starts with n_candidates, keeps top 1/factor each
    # successive halving round, using more data per round.
    # 60 candidates, factor=3 → rounds of 60→20→7 candidates ≈ 5-8x faster
    # than RandomizedSearchCV(n_iter=60).
    _halving_kw = dict(
        factor=3, cv=tscv, scoring='neg_mean_squared_error',
        random_state=42, n_jobs=-1, refit=True,
        min_resources='exhaust',   # use all tune data in the final round
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

    # ── Validation: monthly expanding window ──────────────────────────────────
    # Refits once per calendar month on all data before that month.
    # ~60 fits instead of ~1250 daily fits — predictions still daily.
    rf_val_acts, rf_val_preds = periodic_expanding_window_eval(
        RandomForestRegressor, {**rf_best, 'random_state': 42, 'n_jobs': -1},
        val_clean.index, pretest_clean, target, predictors,
    )
    lgb_val_acts, lgb_val_preds = periodic_expanding_window_eval(
        LGBMRegressor, {**lgb_best, 'random_state': 42, 'n_jobs': -1, 'verbose': -1},
        val_clean.index, pretest_clean, target, predictors,
    )

    pre_val_Y = full_clean.loc[:'2015-12-31', target].values

    rf_val_oos_r2  = expanding_oos_r2(rf_val_acts,  rf_val_preds,  pre_val_Y)
    lgb_val_oos_r2 = expanding_oos_r2(lgb_val_acts, lgb_val_preds, pre_val_Y)
    rf_val_ic      = information_coefficient(rf_val_acts,  rf_val_preds)
    lgb_val_ic     = information_coefficient(lgb_val_acts, lgb_val_preds)

    print(f"\n  [RF]  Val  OOS-R²: {rf_val_oos_r2:.4f}   IC: {rf_val_ic:.4f}")
    print(f"  [LGB] Val  OOS-R²: {lgb_val_oos_r2:.4f}   IC: {lgb_val_ic:.4f}")

    # ── Test: monthly expanding window ────────────────────────────────────────
    rf_test_acts, rf_test_preds = periodic_expanding_window_eval(
        RandomForestRegressor, {**rf_best, 'random_state': 42, 'n_jobs': -1},
        test_clean.index, full_clean, target, predictors,
    )
    lgb_test_acts, lgb_test_preds = periodic_expanding_window_eval(
        LGBMRegressor, {**lgb_best, 'random_state': 42, 'n_jobs': -1, 'verbose': -1},
        test_clean.index, full_clean, target, predictors,
    )

    pre_test_Y = full_clean.loc[:'2020-12-31', target].values

    rf_test_oos_r2  = expanding_oos_r2(rf_test_acts,  rf_test_preds,  pre_test_Y)
    lgb_test_oos_r2 = expanding_oos_r2(lgb_test_acts, lgb_test_preds, pre_test_Y)
    rf_test_mse     = test_mse(rf_test_acts,  rf_test_preds)
    lgb_test_mse    = test_mse(lgb_test_acts, lgb_test_preds)
    rf_test_ic      = information_coefficient(rf_test_acts,  rf_test_preds)
    lgb_test_ic     = information_coefficient(lgb_test_acts, lgb_test_preds)

    print(f"\n  [RF]  Test OOS-R²: {rf_test_oos_r2:.4f}   MSE: {rf_test_mse:.6f}   IC: {rf_test_ic:.4f}")
    print(f"  [LGB] Test OOS-R²: {lgb_test_oos_r2:.4f}   MSE: {lgb_test_mse:.6f}   IC: {lgb_test_ic:.4f}")

    # ── Store results ─────────────────────────────────────────────────────────
    all_cv.setdefault(target, {}).update({
        'rf_cv_mse': rf_cv_mse, 'lgb_cv_mse': lgb_cv_mse,
    })
    all_val.setdefault(target, {}).update({
        'rf_val_oos_r2':  rf_val_oos_r2,  'rf_val_ic':  rf_val_ic,
        'lgb_val_oos_r2': lgb_val_oos_r2, 'lgb_val_ic': lgb_val_ic,
    })
    all_oos.setdefault(target, {}).update({
        'rf_test_oos_r2':  rf_test_oos_r2,  'rf_test_mse':  rf_test_mse,  'rf_test_ic':  rf_test_ic,
        'lgb_test_oos_r2': lgb_test_oos_r2, 'lgb_test_mse': lgb_test_mse, 'lgb_test_ic': lgb_test_ic,
    })
