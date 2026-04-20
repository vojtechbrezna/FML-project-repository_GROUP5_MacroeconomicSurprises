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
from ISLP.models import ModelSpec as MS
from ISLP.models import (Stepwise, sklearn_selected, sklearn_selection_path)

# Ensure Numba (used by l0bnb) has a writable cache dir
_numba_cache_dir = os.path.join(tempfile.gettempdir(), "numba_cache")
os.makedirs(_numba_cache_dir, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", _numba_cache_dir)
from l0bnb import fit_path

# ══════════════════════════════════════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════════════════════════════════════

df = pd.read_csv('test_set.csv', index_col=0, parse_dates=True)
df = df.sort_index()

train_df = df.loc[:'2020-12-31']
test_df  = df.loc['2021-01-01':]

ret_cols  = [c for c in df.columns if c.startswith('ret_') and '_lag' not in c]
surp_cols = [c for c in df.columns if c.startswith('surp_')]

K    = 5
tscv = TimeSeriesSplit(n_splits=K)

print(f"Train: {train_df.index.min().date()} → {train_df.index.max().date()}  ({len(train_df)} obs)")
print(f"Test:  {test_df.index.min().date()} → {test_df.index.max().date()}   ({len(test_df)} obs)")
print(f"Targets: {ret_cols}")
print(f"Predictors per target: 1 AR + {len(surp_cols)} surp + 1 VIX = {1+len(surp_cols)+1}")

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def nCp(sigma2, estimator, X, Y):
    """Negative Cp statistic (higher = better, for sklearn_selected)."""
    n, p = X.shape
    Yhat = estimator.predict(X)
    RSS  = np.sum((Y - Yhat) ** 2)
    return -(RSS + 2 * p * sigma2) / n

def insample_stats(Y, Yhat, p, sigma2):
    n   = len(Y)
    TSS = np.sum((Y - Y.mean()) ** 2)
    RSS = np.sum((Y - Yhat) ** 2)
    adj_r2 = 1 - (RSS / (n - p - 1)) / (TSS / (n - 1))
    cp     = RSS / sigma2 + 2 * p - n
    bic    = n * np.log(RSS / n) + p * np.log(n)
    return adj_r2, cp, bic

def test_mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))

def oos_r2(y_train, y_test, y_pred):
    """Campbell-Thompson OOS R-squared (benchmark = training mean)."""
    ss_res  = np.sum((y_test - y_pred) ** 2)
    ss_null = np.sum((y_test - y_train.mean()) ** 2)
    return float(1.0 - ss_res / ss_null)

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

def _cv_mse_of_vars(vars_list, D, Y, cv):
    """CV MSE of a LinearRegression on the given variable subset (uses D, already standardised)."""
    X_cv = D[vars_list].values if vars_list else np.zeros((len(Y), 1))
    return float(-skm.cross_validate(
        skl.LinearRegression(), X_cv, Y,
        cv=cv, scoring='neg_mean_squared_error')['test_score'].mean())

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

for ti, target in enumerate(ret_cols):

    print(f"\n{'='*70}")
    print(f"  TARGET: {target}")
    print(f"{'='*70}")

    ar_term    = f"{target}_lag1"
    predictors = [ar_term] + surp_cols + ['vix_lag1']

    # ── Clean rows ────────────────────────────────────────────────────────────
    train_clean = train_df[[target] + predictors].dropna()
    test_clean  = test_df[[target]  + predictors].dropna()

    Y_train    = train_clean[target].values
    X_train_df = train_clean[predictors]
    X_train    = X_train_df.values
    Y_test     = test_clean[target].values
    X_test_df  = test_clean[predictors]
    X_test     = X_test_df.values
    n_train    = len(Y_train)

    print(f"  Train obs: {n_train}  |  Test obs: {len(Y_test)}")

    # ── Standardize using training stats only ─────────────────────────────────
    # Used for: sigma2, ISLP stepwise, solution-path plots, subset in-sample stats
    scaler       = StandardScaler().fit(X_train)
    X_train_s    = scaler.transform(X_train)
    X_test_s     = scaler.transform(X_test)
    X_train_s_df = pd.DataFrame(X_train_s, columns=predictors, index=X_train_df.index)
    X_test_s_df  = pd.DataFrame(X_test_s,  columns=predictors, index=X_test_df.index)

    # ── AR(1) benchmark — Pipeline fits scaler per CV fold ────────────────────
    ar1_pipe = Pipeline([('sc', StandardScaler()), ('lr', skl.LinearRegression())])
    ar1_cv_mse = float(-skm.cross_validate(
        ar1_pipe, X_train[:, [0]], Y_train,
        cv=tscv, scoring='neg_mean_squared_error')['test_score'].mean())
    ar1_pipe.fit(X_train[:, [0]], Y_train)
    ar1_test_mse = test_mse(Y_test, ar1_pipe.predict(X_test[:, [0]]))

    # ── sigma2 and ISLP design ────────────────────────────────────────────────
    design           = MS(predictors).fit(X_train_s_df)
    X_with_intercept = design.transform(X_train_s_df)
    sigma2           = OLS(Y_train, X_with_intercept).fit().scale
    neg_Cp           = partial(nCp, sigma2)

    D_train  = design.fit_transform(X_train_s_df).drop('intercept', axis=1)
    D_test   = design.transform(X_test_s_df).drop('intercept', axis=1)
    X_no_int = np.asarray(D_train)

    # ── Per-fold standardisation helper (for ISLP manual CV loops) ───────────
    def _fold_dfs(tr_idx, te_idx):
        fs  = StandardScaler().fit(X_train[tr_idx])
        Xtr = pd.DataFrame(fs.transform(X_train[tr_idx]),
                            columns=predictors, index=X_train_df.index[tr_idx])
        Xte = pd.DataFrame(fs.transform(X_train[te_idx]),
                            columns=predictors, index=X_train_df.index[te_idx])
        return Xtr, Xte

    # ══════════════════════════════════════════════════════════════════════════
    #  SUBSET SELECTION
    # ══════════════════════════════════════════════════════════════════════════

    # -- Forward Cp -----------------------------------------------------------
    fwd_strategy_cp = Stepwise.first_peak(design, direction='forward',
                                           max_terms=len(design.terms))
    forward_Cp = sklearn_selected(OLS, fwd_strategy_cp, scoring=neg_Cp)
    forward_Cp.fit(X_train_s_df, Y_train)
    fwd_cp_vars   = [c for c in D_train.columns if c in forward_Cp.selected_state_]
    fwd_cp_cv_mse = _cv_mse_of_vars(fwd_cp_vars, D_train, Y_train, tscv)

    # -- Forward CV path ------------------------------------------------------
    fwd_strategy_cv = Stepwise.fixed_steps(design, len(design.terms), direction='forward')
    forward_cv_path = sklearn_selection_path(OLS, fwd_strategy_cv)
    forward_cv_path.fit(X_train_s_df, Y_train)
    Yhat_fwd_in     = forward_cv_path.predict(X_train_s_df)   # (n, n_steps)
    n_steps_cv      = len(forward_cv_path.models_)

    cv_fwd_mse = []
    for tr_idx, te_idx in tscv.split(Y_train):
        _Xf_tr, _Xf_te = _fold_dfs(tr_idx, te_idx)
        _path = sklearn_selection_path(OLS, Stepwise.fixed_steps(
            design, len(design.terms), direction='forward'))
        _path.fit(_Xf_tr, Y_train[tr_idx])
        _preds = _path.predict(_Xf_te)
        n_s = min(_preds.shape[1], n_steps_cv)
        fe  = np.full((n_steps_cv,), np.nan)
        fe[:n_s] = ((_preds[:, :n_s] - Y_train[te_idx, None]) ** 2).mean(0)
        cv_fwd_mse.append(fe)
    cv_fwd_mse = np.array(cv_fwd_mse).T   # (n_steps, K)

    best_cv_step  = int(np.nanargmin(np.nanmean(cv_fwd_mse, axis=1)))
    best_cv_state = forward_cv_path.models_[best_cv_step][0]
    fwd_cv_vars   = _state_to_D_columns(best_cv_state, D_train.columns)

    # Validation curve (last 20%, for plot only)
    val_cut = int(n_train * 0.8)
    forward_cv_path.fit(X_train_s_df.iloc[:val_cut], Y_train[:val_cut])
    Yhat_val_fwd = forward_cv_path.predict(X_train_s_df.iloc[val_cut:])
    val_mse_fwd  = ((Yhat_val_fwd - Y_train[val_cut:, None]) ** 2).mean(0)
    forward_cv_path.fit(X_train_s_df, Y_train)   # refit on full training

    # Forward stepwise plot
    ax = axes_fwd[0, ti]
    insample_mse_fwd = ((Yhat_fwd_in - Y_train[:, None]) ** 2).mean(0)
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

    # -- Backward Cp ----------------------------------------------------------
    bwd_strategy_cp = Stepwise.first_peak(design, direction='backward',
                                           max_terms=len(design.terms),
                                           initial_terms=list(design.terms))
    backward_Cp = sklearn_selected(OLS, bwd_strategy_cp, scoring=neg_Cp)
    backward_Cp.fit(X_train_s_df, Y_train)
    bwd_cp_vars   = [c for c in D_train.columns if c in backward_Cp.selected_state_]
    bwd_cp_cv_mse = _cv_mse_of_vars(bwd_cp_vars, D_train, Y_train, tscv)

    # -- Backward CV path -----------------------------------------------------
    bwd_strategy_cv  = Stepwise.fixed_steps(design, len(design.terms), direction='backward',
                                             initial_terms=list(design.terms))
    backward_cv_path = sklearn_selection_path(OLS, bwd_strategy_cv)
    backward_cv_path.fit(X_train_s_df, Y_train)
    Yhat_bwd_in      = backward_cv_path.predict(X_train_s_df)
    n_steps_bwd      = len(backward_cv_path.models_)

    cv_bwd_mse = []
    for tr_idx, te_idx in tscv.split(Y_train):
        _Xf_tr, _Xf_te = _fold_dfs(tr_idx, te_idx)
        _path = sklearn_selection_path(OLS, Stepwise.fixed_steps(
            design, len(design.terms), direction='backward',
            initial_terms=list(design.terms)))
        _path.fit(_Xf_tr, Y_train[tr_idx])
        _preds = _path.predict(_Xf_te)
        n_s = min(_preds.shape[1], n_steps_bwd)
        fe  = np.full((n_steps_bwd,), np.nan)
        fe[:n_s] = ((_preds[:, :n_s] - Y_train[te_idx, None]) ** 2).mean(0)
        cv_bwd_mse.append(fe)
    cv_bwd_mse = np.array(cv_bwd_mse).T   # (n_steps, K)

    best_bwd_step  = int(np.nanargmin(np.nanmean(cv_bwd_mse, axis=1)))
    best_bwd_state = backward_cv_path.models_[best_bwd_step][0]
    bwd_cv_vars    = _state_to_D_columns(best_bwd_state, D_train.columns)

    # Validation curve for backward
    backward_cv_path.fit(X_train_s_df.iloc[:val_cut], Y_train[:val_cut])
    Yhat_val_bwd = backward_cv_path.predict(X_train_s_df.iloc[val_cut:])
    val_mse_bwd  = ((Yhat_val_bwd - Y_train[val_cut:, None]) ** 2).mean(0)
    backward_cv_path.fit(X_train_s_df, Y_train)   # refit on full training

    # Backward stepwise plot
    ax = axes_bwd[0, ti]
    insample_mse_bwd = ((Yhat_bwd_in - Y_train[:, None]) ** 2).mean(0)
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

    # -- In-sample predictions for subset methods -----------------------------
    def _lr_yhat(vars_list, D):
        if not vars_list:
            return np.full(len(Y_train), Y_train.mean())
        return skl.LinearRegression().fit(D[vars_list], Y_train).predict(D[vars_list])

    Yhat_fwd_cp   = _lr_yhat(fwd_cp_vars, D_train)
    Yhat_fwd_cv_t = forward_cv_path.predict(X_train_s_df)[:, best_cv_step]
    Yhat_bwd_cp   = _lr_yhat(bwd_cp_vars, D_train)
    Yhat_bwd_cv_t = backward_cv_path.predict(X_train_s_df)[:, best_bwd_step]

    # -- Best subset selection (l0bnb / Cp) -----------------------------------
    fit_best   = fit_path(X_no_int, Y_train, max_nonzeros=X_no_int.shape[1])
    pred_names = list(D_train.columns)
    coefs_list = [(np.array(s['B']).ravel(), float(s.get('intercept', 0)))
                  for s in fit_best]
    cps = [np.sum((Y_train - X_no_int @ B - b) ** 2) / sigma2
           + 2 * (np.abs(B) > 1e-8).sum() - n_train
           for B, b in coefs_list]
    best_B, _     = coefs_list[int(np.argmin(cps))]
    best_sub_vars = [pred_names[j] for j in np.where(np.abs(best_B) > 1e-8)[0]]
    best_sub_cv_mse = _cv_mse_of_vars(best_sub_vars, D_train, Y_train, tscv)
    Yhat_best_sub   = _lr_yhat(best_sub_vars, D_train)

    # Subset in-sample comparison table
    subset_rows = []
    for name, yh, p_model, cv_val in [
        ('Fwd (Cp)',      Yhat_fwd_cp,   len(fwd_cp_vars),   fwd_cp_cv_mse),
        ('Fwd (CV)',      Yhat_fwd_cv_t, len(fwd_cv_vars),
                          float(np.nanmean(cv_fwd_mse, axis=1)[best_cv_step])),
        ('Bwd (Cp)',      Yhat_bwd_cp,   len(bwd_cp_vars),   bwd_cp_cv_mse),
        ('Bwd (CV)',      Yhat_bwd_cv_t, len(bwd_cv_vars),
                          float(np.nanmean(cv_bwd_mse, axis=1)[best_bwd_step])),
        ('Best Sub (Cp)', Yhat_best_sub, len(best_sub_vars), best_sub_cv_mse),
    ]:
        adj_r2, cp_val, bic = insample_stats(Y_train, yh, p_model, sigma2)
        subset_rows.append({'Method': name, 'p': p_model,
                            'Adj.R2': adj_r2, 'Cp': cp_val, 'BIC': bic, 'CV MSE': cv_val})
    print(f"\n--- Subset selection ({target}) ---")
    print(pd.DataFrame(subset_rows).to_string(index=False, float_format='%.5f'))

    # ══════════════════════════════════════════════════════════════════════════
    #  RIDGE — Pipeline fits scaler per CV fold
    # ══════════════════════════════════════════════════════════════════════════
    lambdas    = 10 ** np.linspace(8, -6, 100) / Y_train.std()
    grid_ridge = skm.GridSearchCV(
        Pipeline([('sc', StandardScaler()), ('ridge', skl.Ridge())]),
        {'ridge__alpha': lambdas}, cv=tscv, scoring='neg_mean_squared_error')
    grid_ridge.fit(X_train, Y_train)
    ridge_alpha  = grid_ridge.best_params_['ridge__alpha']
    ridge_cv_mse = -grid_ridge.best_score_

    # Ridge solution-path plot (X_train_s used for path visualisation only)
    soln_ridge    = skl.ElasticNet.path(X_train_s, Y_train, l1_ratio=0., alphas=lambdas)[1]
    soln_ridge_df = pd.DataFrame(soln_ridge.T, columns=D_train.columns, index=np.log(lambdas))
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

    # ══════════════════════════════════════════════════════════════════════════
    #  LASSO — Pipeline fits scaler per CV fold
    # ══════════════════════════════════════════════════════════════════════════
    lasso_alphas = np.logspace(-6, 2, 100)
    grid_lasso   = skm.GridSearchCV(
        Pipeline([('sc', StandardScaler()), ('lasso', skl.Lasso(max_iter=10000))]),
        {'lasso__alpha': lasso_alphas}, cv=tscv, scoring='neg_mean_squared_error')
    grid_lasso.fit(X_train, Y_train)
    lasso_alpha  = grid_lasso.best_params_['lasso__alpha']
    lasso_cv_mse = -grid_lasso.best_score_
    p_lasso_nz   = int((np.abs(grid_lasso.best_estimator_['lasso'].coef_) > 1e-8).sum())

    # Use l1_ratio=1.0 to match the Lasso estimator (pure L1); sklearn's path helper defaults to elastic-net.
    alphas_path, soln_lasso, _ = skl.Lasso.path(X_train_s, Y_train, alphas=lasso_alphas, l1_ratio=1.0)
    # sklearn returns alphas in decreasing order; align x-axis to the returned path
    if alphas_path[0] > alphas_path[-1]:
        alphas_path = alphas_path[::-1]
        soln_lasso = soln_lasso[:, ::-1]
    soln_lasso_df    = pd.DataFrame(soln_lasso.T, columns=D_train.columns,
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
    }, index=D_train.columns)
    coef_table['Lasso selected'] = coef_table['Lasso'].abs() > 1e-8
    print(f"\n--- Ridge vs Lasso coefficients ({target}) ---")
    print(coef_table.to_string(float_format='%.5f'))

    # ══════════════════════════════════════════════════════════════════════════
    #  PCR — Pipeline fits scaler per CV fold
    # ══════════════════════════════════════════════════════════════════════════
    max_comp = min(50, X_train.shape[1], X_train.shape[0] - 1)
    pipe_pcr = Pipeline([('sc', StandardScaler()), ('pca', PCA()),
                          ('lr', skl.LinearRegression())])
    grid_pcr = skm.GridSearchCV(pipe_pcr, {'pca__n_components': range(1, max_comp + 1)},
                                 cv=tscv, scoring='neg_mean_squared_error')
    grid_pcr.fit(X_train, Y_train)
    pcr_best_n = grid_pcr.best_params_['pca__n_components']
    pcr_cv_mse = -grid_pcr.best_score_

    # ══════════════════════════════════════════════════════════════════════════
    #  PLS — Pipeline fits scaler per CV fold
    # ══════════════════════════════════════════════════════════════════════════
    max_comp_pls = min(20, X_train.shape[1])
    grid_pls = skm.GridSearchCV(
        Pipeline([('sc', StandardScaler()), ('pls', PLSRegression(scale=False))]),
        {'pls__n_components': range(1, max_comp_pls + 1)},
        cv=tscv, scoring='neg_mean_squared_error')
    grid_pls.fit(X_train, Y_train)
    pls_best_n = grid_pls.best_params_['pls__n_components']
    pls_cv_mse = -grid_pls.best_score_

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
    #  RESULTS TABLES
    # ══════════════════════════════════════════════════════════════════════════

    Yhat_ar1   = ar1_pipe.predict(X_train[:, [0]])
    Yhat_ridge = grid_ridge.best_estimator_.predict(X_train)
    Yhat_lasso = grid_lasso.best_estimator_.predict(X_train)
    Yhat_pcr   = grid_pcr.best_estimator_.predict(X_train)
    Yhat_pls   = grid_pls.best_estimator_.predict(X_train).ravel()

    # Table 1: CV MSE
    cv_rows = [
        ('AR(1) benchmark', ar1_cv_mse,   '1 predictor'),
        ('Fwd (Cp)',         fwd_cp_cv_mse, f'{len(fwd_cp_vars)} pred.'),
        ('Fwd (CV)',         float(np.nanmean(cv_fwd_mse, axis=1)[best_cv_step]),
                             f'{best_cv_step} pred.'),
        ('Bwd (Cp)',         bwd_cp_cv_mse, f'{len(bwd_cp_vars)} pred.'),
        ('Bwd (CV)',         float(np.nanmean(cv_bwd_mse, axis=1)[best_bwd_step]),
                             f'{best_bwd_step} removed'),
        ('Best Sub (Cp)',    best_sub_cv_mse, f'{len(best_sub_vars)} pred.'),
        ('Ridge',            ridge_cv_mse,  f'alpha={ridge_alpha:.2e}'),
        ('Lasso',            lasso_cv_mse,  f'alpha={lasso_alpha:.2e}, nz={p_lasso_nz}'),
        ('PCR',              pcr_cv_mse,    f'{pcr_best_n} comp.'),
        ('PLS',              pls_cv_mse,    f'{pls_best_n} comp.'),
    ]
    cv_df = pd.DataFrame(cv_rows, columns=['Method', 'CV MSE (tscv)', 'Best param'])
    all_cv[target] = cv_df
    print(f"\n--- Table 1: CV MSE ({target}) ---")
    print(cv_df.to_string(index=False, float_format='%.6f'))

    # Table 2: in-sample stats
    insam_rows = []
    for name, yh, p_model in [
        ('AR(1) benchmark', Yhat_ar1,      1),
        ('Fwd (Cp)',         Yhat_fwd_cp,   len(fwd_cp_vars)),
        ('Fwd (CV)',         Yhat_fwd_cv_t, len(fwd_cv_vars)),
        ('Bwd (Cp)',         Yhat_bwd_cp,   len(bwd_cp_vars)),
        ('Bwd (CV)',         Yhat_bwd_cv_t, len(bwd_cv_vars)),
        ('Best Sub (Cp)',    Yhat_best_sub, len(best_sub_vars)),
        ('Ridge',            Yhat_ridge,    X_train.shape[1]),
        ('Lasso',            Yhat_lasso,    p_lasso_nz),
        ('PCR',              Yhat_pcr,      pcr_best_n),
        ('PLS',              Yhat_pls,      pls_best_n),
    ]:
        adj_r2, cp_val, bic = insample_stats(Y_train, yh, p_model, sigma2)
        insam_rows.append({'Method': name, 'p': p_model,
                           'Adj.R2': adj_r2, 'Cp': cp_val, 'BIC': bic})
    insam_df = pd.DataFrame(insam_rows)
    all_insam[target] = insam_df
    print(f"\n--- Table 2: In-sample stats ({target}) ---")
    print(insam_df.to_string(index=False, float_format='%.5f'))

    # Table 3: OOS test MSE
    def _lr_test(vars_list, D_tr, D_te):
        if not vars_list:
            return np.full(len(Y_test), Y_train.mean())
        m = skl.LinearRegression().fit(D_tr[vars_list], Y_train)
        return m.predict(D_te[vars_list])

    _yhat_fwd_cp   = _lr_test(fwd_cp_vars,   D_train, D_test)
    _yhat_fwd_cv   = forward_cv_path.predict(X_test_s_df)[:, best_cv_step]
    _yhat_bwd_cp   = _lr_test(bwd_cp_vars,   D_train, D_test)
    _yhat_bwd_cv   = backward_cv_path.predict(X_test_s_df)[:, best_bwd_step]
    _yhat_best_sub = _lr_test(best_sub_vars,  D_train, D_test)
    _yhat_ridge    = grid_ridge.best_estimator_.predict(X_test)
    _yhat_lasso    = grid_lasso.best_estimator_.predict(X_test)
    _yhat_pcr      = grid_pcr.best_estimator_.predict(X_test)
    _yhat_pls      = grid_pls.best_estimator_.predict(X_test).ravel()
    _yhat_ar1_test = ar1_pipe.predict(X_test[:, [0]])

    oos_rows = [
        ('AR(1) benchmark', ar1_test_mse,
         oos_r2(Y_train, Y_test, _yhat_ar1_test)),
        ('Fwd (Cp)',        test_mse(Y_test, _yhat_fwd_cp),
         oos_r2(Y_train, Y_test, _yhat_fwd_cp)),
        ('Fwd (CV)',        test_mse(Y_test, _yhat_fwd_cv),
         oos_r2(Y_train, Y_test, _yhat_fwd_cv)),
        ('Bwd (Cp)',        test_mse(Y_test, _yhat_bwd_cp),
         oos_r2(Y_train, Y_test, _yhat_bwd_cp)),
        ('Bwd (CV)',        test_mse(Y_test, _yhat_bwd_cv),
         oos_r2(Y_train, Y_test, _yhat_bwd_cv)),
        ('Best Sub (Cp)',   test_mse(Y_test, _yhat_best_sub),
         oos_r2(Y_train, Y_test, _yhat_best_sub)),
        ('Ridge',           test_mse(Y_test, _yhat_ridge),
         oos_r2(Y_train, Y_test, _yhat_ridge)),
        ('Lasso',           test_mse(Y_test, _yhat_lasso),
         oos_r2(Y_train, Y_test, _yhat_lasso)),
        ('PCR',             test_mse(Y_test, _yhat_pcr),
         oos_r2(Y_train, Y_test, _yhat_pcr)),
        ('PLS',             test_mse(Y_test, _yhat_pls),
         oos_r2(Y_train, Y_test, _yhat_pls)),
    ]
    oos_df = pd.DataFrame(oos_rows, columns=['Method', 'Test MSE', 'OOS R2'])
    all_oos[target] = oos_df
    print(f"\n--- Table 3: Out-of-sample ({target}) ---")
    print(oos_df.to_string(index=False, float_format='%.6f'))

    # OOS R² bar chart
    ax = axes_oos[0, ti]
    bar_colors = ['gold' if v == oos_df['OOS R2'].max() else 'steelblue'
                  for v in oos_df['OOS R2']]
    bars = ax.bar(oos_df['Method'], oos_df['OOS R2'], color=bar_colors)
    ax.bar_label(bars, fmt='%.4f', fontsize=7, padding=2)
    ax.set_ylabel('OOS R²', fontsize=11)
    ax.set_title(f'OOS R² (Campbell-Thompson) — {target}', fontsize=11)
    ax.set_xticklabels(oos_df['Method'], rotation=35, ha='right', fontsize=8)
    _r2_min = oos_df['OOS R2'].min(); _r2_max = oos_df['OOS R2'].max()
    ax.set_ylim(_r2_min - 0.1 * abs(_r2_min) - 0.001,
                _r2_max + 0.1 * abs(_r2_max) + 0.001)

# ══════════════════════════════════════════════════════════════════════════════
#  SAVE SUMMARY FIGURES
# ══════════════════════════════════════════════════════════════════════════════

_summary_plots = [
    (fig_fwd,   'fwd_stepwise_summary.png'),
    (fig_bwd,   'bwd_stepwise_summary.png'),
    (fig_rpath, 'ridge_path_summary.png'),
    (fig_rcv,   'ridge_cv_summary.png'),
    (fig_lpath, 'lasso_path_summary.png'),
    (fig_lcv,   'lasso_cv_summary.png'),
    (fig_dim,   'pcr_pls_summary.png'),
    (fig_oos,   'oos_r2_summary.png'),
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
    {t: r.set_index('Method')['Test MSE'] for t, r in all_oos.items()}, axis=1
)
summary_oos_r2 = pd.concat(
    {t: r.set_index('Method')['OOS R2'] for t, r in all_oos.items()}, axis=1
)
summary_cv = pd.concat(
    {t: r.set_index('Method')['CV MSE (tscv)'] for t, r in all_cv.items()}, axis=1
)

print("\n" + "="*70)
print("CROSS-ASSET SUMMARY — Test MSE (2021–2025)")
print("="*70)
print(summary_oos_mse.to_string(float_format='%.6f'))

print("\n" + "="*70)
print("CROSS-ASSET SUMMARY — OOS R² (Campbell-Thompson, 2021–2025)")
print("="*70)
print(summary_oos_r2.to_string(float_format='%.6f'))

print("\n" + "="*70)
print("CROSS-ASSET SUMMARY — CV MSE (TimeSeriesSplit, train only)")
print("="*70)
print(summary_cv.to_string(float_format='%.6f'))

summary_oos_mse.to_csv('model_selection_results_oos.csv')
summary_oos_r2.to_csv('model_selection_results_oos_r2.csv')
summary_cv.to_csv('model_selection_results_cv.csv')
print("\nSaved: model_selection_results_oos.csv, model_selection_results_oos_r2.csv, model_selection_results_cv.csv")
