# -*- coding: utf-8 -*-
"""
baselines.py
============
All baseline methods compared against FDR. Implementations are shared
across both evaluation scenarios (CV and OOD) -- only the train/test split
differs (handled in run_experiments.py), not the method logic.

Groups (see config.METHOD_META for labels used in tables/plots):
  A. Classical regressors wrapped with TP-toggling counterfactual prediction
     (CatBoost, Neural Net, Random Forest, SVR, XGBoost, Linear Regression)
  B. Meta-learners (S-, X-, DR-, R-Learner)
  C. Causal Forest (DML)
  D. Modern SOTA (CUTS, EP-Learner, BITES)

NOTE: TARNet and DragonNet have been intentionally removed from this baseline
set per current experimental scope.
"""

import numpy as np
import pandas as pd
import torch

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

import config


# ============================================================
# Shared counterfactual-recommendation helper
# ============================================================

def recommend_treatment(model, X_test: pd.DataFrame, treatment_plans: list) -> pd.DataFrame:
    """
    Generate counterfactual predictions for each treatment arm by toggling
    the TP indicator columns, for any joint model exposing .predict().
    Recommends the arm with the lowest predicted outcome (lower RCB = better).
    """
    predicted_outcomes = pd.DataFrame(index=X_test.index)
    for tp in treatment_plans:
        X_cf = X_test.copy()
        X_cf[treatment_plans] = 0
        X_cf[tp] = 1
        predicted_outcomes[tp] = model.predict(X_cf)
    predicted_outcomes["REC_TP"] = predicted_outcomes.idxmin(axis=1)
    return predicted_outcomes


def _wrap_sklearn(model_factory):
    """Wrap a sklearn-compatible model: fit -> recommend_treatment via TP toggling."""
    def recommend_fn(X_train, y_train, X_test, treatment_plans):
        m = model_factory()
        m.fit(X_train, y_train)
        return recommend_treatment(m, X_test, treatment_plans)
    return recommend_fn


# ============================================================
# A. Classical baselines
# ============================================================

def _classical_models():
    # Reduced to CatBoost and XGBoost only (Neural Net, Random Forest, SVR,
    # and Linear Regression removed from the exposed baseline set -- see
    # config.METHOD_META). RandomForestRegressor is still used internally
    # elsewhere in this file (recommend_rsf_tlearner's Causal Forest
    # fallback), so that import is kept.
    return {
        "CB":  _wrap_sklearn(lambda: CatBoostRegressor(verbose=0, random_state=config.SEED)),
        "XGB": _wrap_sklearn(lambda: XGBRegressor(random_state=config.SEED, verbosity=0)),
    }


# ============================================================
# B. Meta-learner baselines
# ============================================================

def recommend_s_learner(X_train, y_train, X_test, treatment_plans):
    """S-Learner: single joint model, TP columns as features."""
    model = GradientBoostingRegressor(n_estimators=100, random_state=config.SEED)
    model.fit(X_train, y_train)
    return recommend_treatment(model, X_test, treatment_plans)


def recommend_t_learner_gb(X_train, y_train, X_test, treatment_plans):
    """T-Learner (GradientBoosting variant): one model per arm. Used as a
    generic fallback for methods whose optional dependency is unavailable."""
    predicted_outcomes = pd.DataFrame(index=X_test.index)
    feature_cols = [c for c in X_train.columns if c not in treatment_plans]
    for tp in treatment_plans:
        mask = X_train[tp] == 1
        if mask.sum() < 5:
            predicted_outcomes[tp] = y_train.mean()
            continue
        model = GradientBoostingRegressor(n_estimators=100, random_state=config.SEED)
        model.fit(X_train.loc[mask, feature_cols], y_train[mask])
        predicted_outcomes[tp] = model.predict(X_test[feature_cols])
    predicted_outcomes["REC_TP"] = predicted_outcomes.idxmin(axis=1)
    return predicted_outcomes


def recommend_rsf_tlearner(X_train, y_train, X_test, treatment_plans):
    """T-Learner (RandomForest variant). Used as a fallback for Causal Forest
    when econml is unavailable."""
    predicted_outcomes = pd.DataFrame(index=X_test.index)
    feature_cols = [c for c in X_train.columns if c not in treatment_plans]
    for tp in treatment_plans:
        mask = X_train[tp] == 1
        if mask.sum() < 5:
            predicted_outcomes[tp] = y_train.mean()
            continue
        model = RandomForestRegressor(n_estimators=100, random_state=config.SEED)
        model.fit(X_train.loc[mask, feature_cols], y_train[mask])
        predicted_outcomes[tp] = model.predict(X_test[feature_cols])
    predicted_outcomes["REC_TP"] = predicted_outcomes.idxmin(axis=1)
    return predicted_outcomes


def recommend_x_learner(X_train, y_train, X_test, treatment_plans):
    """X-Learner (Kunzel et al., 2019): pseudo-effect CATE with propensity blending."""
    feature_cols = [c for c in X_train.columns if c not in treatment_plans]
    X_tr = X_train[feature_cols].values
    X_te = X_test[feature_cols].values

    mu_models, arm_masks = {}, {}
    for tp in treatment_plans:
        mask = X_train[tp].values == 1
        arm_masks[tp] = mask
        if mask.sum() < 5:
            mu_models[tp] = None
            continue
        m = GradientBoostingRegressor(n_estimators=100, random_state=config.SEED)
        m.fit(X_tr[mask], y_train.values[mask])
        mu_models[tp] = m

    cate_models = {}
    for tp in treatment_plans:
        mask = arm_masks[tp]
        if mu_models[tp] is None or mask.sum() < 5:
            cate_models[tp] = None
            continue
        others = [mu_models[o].predict(X_tr[mask])
                  for o, m in mu_models.items() if o != tp and m is not None]
        pseudo = (y_train.values[mask] - np.mean(others, axis=0)
                  if others else np.zeros(mask.sum()))
        cate_m = Ridge(alpha=1.0)
        cate_m.fit(X_tr[mask], pseudo)
        cate_models[tp] = cate_m

    baseline = np.mean([m.predict(X_te) for m in mu_models.values() if m is not None], axis=0)
    predicted_outcomes = pd.DataFrame(index=X_test.index)
    for tp in treatment_plans:
        if cate_models.get(tp) is not None:
            predicted_outcomes[tp] = baseline + cate_models[tp].predict(X_te)
        elif mu_models.get(tp) is not None:
            predicted_outcomes[tp] = mu_models[tp].predict(X_te)
        else:
            predicted_outcomes[tp] = float(y_train.mean())
    predicted_outcomes["REC_TP"] = predicted_outcomes.idxmin(axis=1)
    return predicted_outcomes


def _fit_propensity(X_train, treatment_plans):
    t_label = np.array([
        treatment_plans.index(tp)
        for tp in treatment_plans
        for _ in range((X_train[tp] == 1).sum())
    ])
    X_re = np.vstack([
        X_train[[c for c in X_train.columns if c not in treatment_plans]].values[X_train[tp] == 1]
        for tp in treatment_plans
    ])
    prop = LogisticRegression(max_iter=500, random_state=config.SEED)
    prop.fit(X_re, t_label)
    return prop, treatment_plans


def recommend_dr_learner(X_train, y_train, X_test, treatment_plans):
    """DR-Learner (Kennedy 2020): doubly-robust pseudo-outcomes."""
    feature_cols = [c for c in X_train.columns if c not in treatment_plans]
    X_tr = X_train[feature_cols].values
    X_te = X_test[feature_cols].values
    y_tr = y_train.values

    outcome_model = GradientBoostingRegressor(n_estimators=100, random_state=config.SEED)
    outcome_model.fit(X_train.values, y_tr)

    prop_model, _ = _fit_propensity(X_train, treatment_plans)
    prop_proba = prop_model.predict_proba(X_tr)

    predicted_outcomes = pd.DataFrame(index=X_test.index)
    for i, tp in enumerate(treatment_plans):
        mask = X_train[tp].values == 1
        pi = np.clip(prop_proba[:, i], 1e-6, 1 - 1e-6)
        X_cf = X_train.copy()
        X_cf[treatment_plans] = 0
        X_cf[tp] = 1
        mu_tp = outcome_model.predict(X_cf.values)
        dr_pseudo = mu_tp + mask.astype(float) * (y_tr - mu_tp) / pi
        cate_m = Ridge(alpha=1.0)
        cate_m.fit(X_tr, dr_pseudo)
        predicted_outcomes[tp] = cate_m.predict(X_te)
    predicted_outcomes["REC_TP"] = predicted_outcomes.idxmin(axis=1)
    return predicted_outcomes


def recommend_r_learner(X_train, y_train, X_test, treatment_plans):
    """R-Learner (Nie & Wager 2021): residual-on-residual CATE."""
    feature_cols = [c for c in X_train.columns if c not in treatment_plans]
    X_tr = X_train[feature_cols].values
    X_te = X_test[feature_cols].values
    y_tr = y_train.values

    m_model = GradientBoostingRegressor(n_estimators=100, random_state=config.SEED)
    m_model.fit(X_tr, y_tr)
    y_resid = y_tr - m_model.predict(X_tr)
    baseline_te = m_model.predict(X_te)

    predicted_outcomes = pd.DataFrame(index=X_test.index)
    for tp in treatment_plans:
        t = X_train[tp].values.astype(float)
        e_model = LogisticRegression(max_iter=500, random_state=config.SEED)
        e_model.fit(X_tr, t.astype(int))
        e_hat = np.clip(e_model.predict_proba(X_tr)[:, 1], 1e-6, 1 - 1e-6)
        t_resid = t - e_hat
        safe = np.abs(t_resid) > 1e-4
        if safe.sum() < 5:
            predicted_outcomes[tp] = baseline_te
            continue
        tau_m = Ridge(alpha=1.0)
        tau_m.fit(X_tr[safe], y_resid[safe] / t_resid[safe],
                  sample_weight=t_resid[safe] ** 2)
        predicted_outcomes[tp] = baseline_te + tau_m.predict(X_te)
    predicted_outcomes["REC_TP"] = predicted_outcomes.idxmin(axis=1)
    return predicted_outcomes


# ============================================================
# C. Causal baselines
# ============================================================

def recommend_causal_forest(X_train, y_train, X_test, treatment_plans):
    """Causal Forest (DML). Falls back to a RandomForest T-Learner if econml
    is unavailable or fitting fails for an arm."""
    feature_cols = [c for c in X_train.columns if c not in treatment_plans]
    X_tr = X_train[feature_cols].values
    X_te = X_test[feature_cols].values
    y_tr = y_train.values

    try:
        from econml.dml import CausalForestDML
    except ImportError:
        return recommend_rsf_tlearner(X_train, y_train, X_test, treatment_plans)

    predicted_outcomes = pd.DataFrame(index=X_test.index)
    baseline_model = GradientBoostingRegressor(n_estimators=100, random_state=config.SEED)
    baseline_model.fit(X_tr, y_tr)
    baseline_te = baseline_model.predict(X_te)

    for tp in treatment_plans:
        t = X_train[tp].values.astype(float)
        if t.sum() < 5 or (1 - t).sum() < 5:
            predicted_outcomes[tp] = baseline_te
            continue
        try:
            cf = CausalForestDML(n_estimators=100, random_state=config.SEED, discrete_treatment=False)
            cf.fit(y_tr, t, X=X_tr, W=None)
            predicted_outcomes[tp] = baseline_te + cf.effect(X_te)
        except Exception:
            predicted_outcomes[tp] = baseline_te

    predicted_outcomes["REC_TP"] = predicted_outcomes.idxmin(axis=1)
    return predicted_outcomes


# ============================================================
# D. Modern SOTA baselines
# ============================================================

def recommend_bites(X_train, y_train, X_test, treatment_plans):
    """BITES: MMD-balanced representation network with arm-specific heads."""
    try:
        import torch.nn as nn
    except ImportError:
        return recommend_t_learner_gb(X_train, y_train, X_test, treatment_plans)

    feature_cols = [c for c in X_train.columns if c not in treatment_plans]
    X_np = X_train[feature_cols].values.astype(np.float32)
    y_np = y_train.values.astype(np.float32)
    t_np = np.argmax(X_train[treatment_plans].values, axis=1).astype(np.int64)
    X_te_np = X_test[feature_cols].values.astype(np.float32)
    n_arms, in_dim = len(treatment_plans), X_np.shape[1]

    class BITES_Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.repr = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(),
                                       nn.Linear(64, 32), nn.ReLU())
            self.heads = nn.ModuleList([nn.Linear(32, 1) for _ in range(n_arms)])

        def forward(self, x):
            z = self.repr(x)
            return z, [h(z).squeeze(1) for h in self.heads]

    def mmd_rbf(z1, z2, gamma=1.0):
        def rbf(a, b):
            return torch.exp(-gamma * torch.cdist(a, b) ** 2)
        return rbf(z1, z1).mean() - 2 * rbf(z1, z2).mean() + rbf(z2, z2).mean()

    model = BITES_Net()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    X_t, y_t, t_t = torch.tensor(X_np), torch.tensor(y_np), torch.tensor(t_np)

    for _ in range(60):
        opt.zero_grad()
        z, outs = model(X_t)
        outcome_loss = sum(
            mse(outs[k][t_t == k], y_t[t_t == k])
            for k in range(n_arms) if (t_t == k).any()
        )
        arm_reprs = [z[t_t == k] for k in range(n_arms) if (t_t == k).any()]
        mmd_loss = sum(
            mmd_rbf(arm_reprs[i], arm_reprs[j])
            for i in range(len(arm_reprs))
            for j in range(i + 1, len(arm_reprs))
        )
        (outcome_loss + 0.01 * mmd_loss).backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        _, outs_te = model(torch.tensor(X_te_np))

    predicted_outcomes = pd.DataFrame(index=X_test.index)
    for i, tp in enumerate(treatment_plans):
        predicted_outcomes[tp] = outs_te[i].numpy()
    predicted_outcomes["REC_TP"] = predicted_outcomes.idxmin(axis=1)
    return predicted_outcomes


def recommend_cuts(X_train, y_train, X_test, treatment_plans):
    """CUTS: conformal uncertainty-conservative treatment selection.
    Recommends the arm with the lowest upper-confidence-bound prediction
    (predicted outcome + 90th-percentile residual), rather than just the
    lowest point estimate."""
    feature_cols = [c for c in X_train.columns if c not in treatment_plans]
    X_tr = X_train[feature_cols].values
    X_te = X_test[feature_cols].values
    y_tr = y_train.values

    predicted_outcomes = pd.DataFrame(index=X_test.index)
    uncertainty = pd.DataFrame(index=X_test.index)
    for tp in treatment_plans:
        mask = X_train[tp].values == 1
        if mask.sum() < 5:
            predicted_outcomes[tp] = y_tr.mean()
            uncertainty[tp] = y_tr.std()
            continue
        model = GradientBoostingRegressor(n_estimators=100, random_state=config.SEED)
        model.fit(X_tr[mask], y_tr[mask])
        residuals = np.abs(y_tr[mask] - model.predict(X_tr[mask]))
        predicted_outcomes[tp] = model.predict(X_te)
        uncertainty[tp] = np.quantile(residuals, 0.9)
    ucb = predicted_outcomes[treatment_plans] + uncertainty[treatment_plans]
    predicted_outcomes["REC_TP"] = ucb.idxmin(axis=1)
    return predicted_outcomes


def recommend_ep_learner(X_train, y_train, X_test, treatment_plans):
    """EP-Learner (Curth & van der Schaar 2023): efficient influence function."""
    feature_cols = [c for c in X_train.columns if c not in treatment_plans]
    X_tr = X_train[feature_cols].values
    X_te = X_test[feature_cols].values
    y_tr = y_train.values

    t_np = np.argmax(X_train[treatment_plans].values, axis=1).astype(int)
    prop_model = LogisticRegression(max_iter=500, random_state=config.SEED)
    prop_model.fit(X_tr, t_np)
    prop_proba = prop_model.predict_proba(X_tr)

    mu_models = {}
    for k, tp in enumerate(treatment_plans):
        mask = X_train[tp].values == 1
        if mask.sum() < 5:
            mu_models[tp] = None
            continue
        m = GradientBoostingRegressor(n_estimators=100, random_state=config.SEED)
        m.fit(X_tr[mask], y_tr[mask])
        mu_models[tp] = m

    predicted_outcomes = pd.DataFrame(index=X_test.index)
    for k, tp in enumerate(treatment_plans):
        if mu_models[tp] is None:
            predicted_outcomes[tp] = y_tr.mean()
            continue
        mu_k = mu_models[tp].predict(X_tr)
        e_k = np.clip(prop_proba[:, k], 1e-6, 1.0)
        ind_k = (X_train[tp].values == 1).astype(float)
        eif_k = mu_k + (ind_k / e_k) * (y_tr - mu_k)
        final_model = GradientBoostingRegressor(n_estimators=100, random_state=config.SEED)
        final_model.fit(X_tr, eif_k)
        predicted_outcomes[tp] = final_model.predict(X_te)
    predicted_outcomes["REC_TP"] = predicted_outcomes.idxmin(axis=1)
    return predicted_outcomes


# ============================================================
# Registry (FDR is intentionally NOT included here -- see fdr.py /
# run_experiments.py, which merges {"FDR": ...} with BASELINE_METHODS)
# ============================================================

def get_baseline_methods() -> dict:
    """
    Returns the full baseline method registry (everything except FDR):
    9 baselines across four families -- A. Classical (CatBoost, XGBoost),
    B. Meta-Learner (S/X/DR/R-Learner), C. Causal (Causal Forest), and
    D. Modern SOTA (CUTS, BITES). TARNet, DragonNet, and EP-Learner are
    intentionally excluded from this set (recommend_ep_learner remains
    defined above but is not wired in, in case it is reinstated later).
    """
    methods = dict(_classical_models())
    methods.update({
        "S_L":   recommend_s_learner,
        "X_L":   recommend_x_learner,
        "DR_L":  recommend_dr_learner,
        "R_L":   recommend_r_learner,
        "CF":    recommend_causal_forest,
        "CUTS":  recommend_cuts,
        "BITES": recommend_bites,
    })
    return methods


BASELINE_METHODS = get_baseline_methods()
