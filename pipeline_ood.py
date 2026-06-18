# -*- coding: utf-8 -*-
import os
import warnings
import numpy as np
import pandas as pd

# ============================================================
# Block ALL HuggingFace / telemetry network calls BEFORE
# importing tabpfn -- must be set before the tabpfn import.
# ============================================================
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TABPFN_NO_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from tabpfn import TabPFNRegressor

warnings.filterwarnings("ignore")

SEED = 42
TABPFN_CKPT_PATH = "/scratch/sq95/tv9849/FDR/tabpfn/tabpfn-v3-regressor-v3_default.ckpt"


# ============================================================
# Startup checks
# ============================================================

def _check_ckpt():
    if not os.path.isfile(TABPFN_CKPT_PATH):
        raise FileNotFoundError(
            f"TabPFN checkpoint not found: '{TABPFN_CKPT_PATH}'\n"
            "Download it to that path before running."
        )
    if not os.access(TABPFN_CKPT_PATH, os.R_OK):
        raise PermissionError(f"TabPFN checkpoint not readable: '{TABPFN_CKPT_PATH}'")
    size_mb = os.path.getsize(TABPFN_CKPT_PATH) / 1024 / 1024
    print(f"  [INFO] TabPFN checkpoint OK: '{TABPFN_CKPT_PATH}' ({size_mb:.0f} MB)")


def _check_cuda():
    if not torch.cuda.is_available():
        print("  [INFO] No GPU detected, using CPU.")
        return "cpu"
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    print(f"  [INFO] GPU: {name}, compute capability: sm_{cap[0]}{cap[1]}")
    ver_parts = torch.__version__.split("+")[0].split(".")
    torch_major, torch_minor = int(ver_parts[0]), int(ver_parts[1])
    if cap == (7, 0) and (torch_major, torch_minor) >= (2, 6):
        print(
            f"  [WARN] sm_70 (V100) support dropped in torch 2.6+. "
            f"You have torch {torch.__version__}. Falling back to CPU."
        )
        return "cpu"
    print(f"  [INFO] CUDA sm_{cap[0]}{cap[1]} supported by torch {torch.__version__} -- GPU enabled.")
    return "cuda"


_check_ckpt()
_TABPFN_DEVICE = _check_cuda()
os.environ["TABPFN_DEVICE"] = _TABPFN_DEVICE


# ============================================================
# Utility helpers
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_dataset(path, encoding="ISO-8859-1"):
    return pd.read_csv(path, encoding=encoding, engine="python")


def preprocess_data(dataset, outcome_col, remove_cols, treatment_plans=None):
    """TP indicator columns are NOT scaled -- see pipeline.py for full rationale."""
    drop = [c for c in remove_cols + [outcome_col] if c in dataset.columns]
    X = dataset.drop(columns=drop)
    y = dataset[outcome_col].astype(float)
    valid_idx = y.notna()
    X, y = X.loc[valid_idx], y[valid_idx]
    X = X.dropna(axis=1, how="all")
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = pd.factorize(X[col])[0]
    tp_cols_in_X = [tp for tp in (treatment_plans or []) if tp in X.columns]
    X_tp    = X[tp_cols_in_X].copy()
    X_other = X.drop(columns=tp_cols_in_X)
    imputer = SimpleImputer(strategy="mean")
    X_other_imp = imputer.fit_transform(X_other)
    scaler = StandardScaler()
    X_other_scaled = scaler.fit_transform(X_other_imp)
    X_other_df = pd.DataFrame(X_other_scaled, columns=X_other.columns, index=X.index)
    return pd.concat([X_other_df, X_tp], axis=1), y


def evaluate_model(model, X_test, y_test):
    y_pred = np.array(model.predict(X_test)).ravel()
    return (
        mean_squared_error(y_test, y_pred),
        mean_absolute_error(y_test, y_pred),
        r2_score(y_test, y_pred),
        y_pred,
    )


def recommend_treatment(model, X_test, treatment_plans):
    predicted_outcomes = pd.DataFrame(index=X_test.index)
    for tp in treatment_plans:
        X_cf = X_test.copy()
        X_cf[treatment_plans] = 0
        X_cf[tp] = 1
        predicted_outcomes[tp] = model.predict(X_cf)
    predicted_outcomes["REC_TP"] = predicted_outcomes.idxmin(axis=1)
    return predicted_outcomes


def compare_recommendations(recommended_df, original_df, outcome_col, tp_cols):
    recommended_df = recommended_df.copy()
    recommended_df["CURRENT_TP"] = original_df[tp_cols].idxmax(axis=1).values
    recommended_df["FOLLOW_REC"] = recommended_df["REC_TP"] == recommended_df["CURRENT_TP"]
    combined = pd.concat(
        [original_df.reset_index(drop=True), recommended_df.reset_index(drop=True)],
        axis=1,
    )
    return combined, None, None


def save_summary_stats(summary_list, method_name, final_df, outcome_col):
    if "resp.pCR" in final_df.columns:
        rec_stats = final_df.groupby("FOLLOW_REC")["resp.pCR"].agg(["count", "sum"]).reset_index()
        for _, row in rec_stats.iterrows():
            summary_list["recovery"].append({
                "Method": method_name,
                "FOLLOW_REC": int(row["FOLLOW_REC"]),
                "Count": int(row["count"]),
                "Recovery": int(row["sum"]),
            })
    rcb_stats = final_df.groupby("FOLLOW_REC")[outcome_col].agg(["count", "mean"]).reset_index()
    for _, row in rcb_stats.iterrows():
        summary_list["rcb"].append({
            "Method": method_name,
            "FOLLOW_REC": int(row["FOLLOW_REC"]),
            "Count": int(row["count"]),
            "Avg_RCB_Score": round(row["mean"], 4),
        })


# ============================================================
# Causal / meta-learner baseline methods
# (same implementations as pipeline.py)
# ============================================================

def recommend_treatment_fn(model, X_test, treatment_plans):
    return recommend_treatment(model, X_test, treatment_plans)


def recommend_xgb_joint(X_train, y_train, X_test, treatment_plans):
    model = XGBRegressor(random_state=SEED, n_estimators=100, verbosity=0)
    model.fit(X_train, y_train)
    return recommend_treatment(model, X_test, treatment_plans)


def recommend_rsf_tlearner(X_train, y_train, X_test, treatment_plans):
    predicted_outcomes = pd.DataFrame(index=X_test.index)
    feature_cols = [c for c in X_train.columns if c not in treatment_plans]
    for tp in treatment_plans:
        mask = X_train[tp] == 1
        if mask.sum() < 5:
            predicted_outcomes[tp] = y_train.mean()
            continue
        model = RandomForestRegressor(n_estimators=100, random_state=SEED)
        model.fit(X_train.loc[mask, feature_cols], y_train[mask])
        predicted_outcomes[tp] = model.predict(X_test[feature_cols])
    predicted_outcomes["REC_TP"] = predicted_outcomes.idxmin(axis=1)
    return predicted_outcomes


def recommend_s_learner(X_train, y_train, X_test, treatment_plans):
    model = GradientBoostingRegressor(n_estimators=100, random_state=SEED)
    model.fit(X_train, y_train)
    return recommend_treatment(model, X_test, treatment_plans)


def recommend_t_learner_gb(X_train, y_train, X_test, treatment_plans):
    predicted_outcomes = pd.DataFrame(index=X_test.index)
    feature_cols = [c for c in X_train.columns if c not in treatment_plans]
    for tp in treatment_plans:
        mask = X_train[tp] == 1
        if mask.sum() < 5:
            predicted_outcomes[tp] = y_train.mean()
            continue
        model = GradientBoostingRegressor(n_estimators=100, random_state=SEED)
        model.fit(X_train.loc[mask, feature_cols], y_train[mask])
        predicted_outcomes[tp] = model.predict(X_test[feature_cols])
    predicted_outcomes["REC_TP"] = predicted_outcomes.idxmin(axis=1)
    return predicted_outcomes


def recommend_causal_forest(X_train, y_train, X_test, treatment_plans):
    feature_cols = [c for c in X_train.columns if c not in treatment_plans]
    X_tr = X_train[feature_cols].values
    X_te = X_test[feature_cols].values
    y_tr = y_train.values
    try:
        from econml.dml import CausalForestDML
    except ImportError:
        return recommend_rsf_tlearner(X_train, y_train, X_test, treatment_plans)
    predicted_outcomes = pd.DataFrame(index=X_test.index)
    baseline_model = GradientBoostingRegressor(n_estimators=100, random_state=SEED)
    baseline_model.fit(X_tr, y_tr)
    baseline_te = baseline_model.predict(X_te)
    for tp in treatment_plans:
        t = X_train[tp].values.astype(float)
        if t.sum() < 5 or (1 - t).sum() < 5:
            predicted_outcomes[tp] = baseline_te
            continue
        try:
            cf = CausalForestDML(n_estimators=100, random_state=SEED, discrete_treatment=False)
            cf.fit(y_tr, t, X=X_tr, W=None)
            predicted_outcomes[tp] = baseline_te + cf.effect(X_te)
        except Exception:
            predicted_outcomes[tp] = baseline_te
    predicted_outcomes["REC_TP"] = predicted_outcomes.idxmin(axis=1)
    return predicted_outcomes



# ============================================================
# Additional baseline methods (matching pipeline.py)
# ============================================================

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
        m = GradientBoostingRegressor(n_estimators=100, random_state=SEED)
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
    from sklearn.linear_model import LogisticRegression
    t_label = np.array([
        treatment_plans.index(tp)
        for tp in treatment_plans
        for _ in range((X_train[tp] == 1).sum())
    ])
    X_re = np.vstack([
        X_train[[c for c in X_train.columns if c not in treatment_plans]].values[X_train[tp] == 1]
        for tp in treatment_plans
    ])
    prop = LogisticRegression(max_iter=500, random_state=SEED)
    prop.fit(X_re, t_label)
    return prop, treatment_plans


def recommend_dr_learner(X_train, y_train, X_test, treatment_plans):
    """DR-Learner (Kennedy 2020): doubly-robust pseudo-outcomes."""
    feature_cols = [c for c in X_train.columns if c not in treatment_plans]
    X_tr = X_train[feature_cols].values
    X_te = X_test[feature_cols].values
    y_tr = y_train.values

    outcome_model = GradientBoostingRegressor(n_estimators=100, random_state=SEED)
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
    from sklearn.linear_model import LogisticRegression
    feature_cols = [c for c in X_train.columns if c not in treatment_plans]
    X_tr = X_train[feature_cols].values
    X_te = X_test[feature_cols].values
    y_tr = y_train.values

    m_model = GradientBoostingRegressor(n_estimators=100, random_state=SEED)
    m_model.fit(X_tr, y_tr)
    y_resid = y_tr - m_model.predict(X_tr)
    baseline_te = m_model.predict(X_te)

    predicted_outcomes = pd.DataFrame(index=X_test.index)
    for tp in treatment_plans:
        t = X_train[tp].values.astype(float)
        e_model = LogisticRegression(max_iter=500, random_state=SEED)
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


def recommend_cuts(X_train, y_train, X_test, treatment_plans):
    """CUTS: conformal uncertainty-conservative treatment selection."""
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
        model = GradientBoostingRegressor(n_estimators=100, random_state=SEED)
        model.fit(X_tr[mask], y_tr[mask])
        residuals = np.abs(y_tr[mask] - model.predict(X_tr[mask]))
        predicted_outcomes[tp] = model.predict(X_te)
        uncertainty[tp] = np.quantile(residuals, 0.9)
    ucb = predicted_outcomes[treatment_plans] + uncertainty[treatment_plans]
    predicted_outcomes["REC_TP"] = ucb.idxmin(axis=1)
    return predicted_outcomes


def recommend_ep_learner(X_train, y_train, X_test, treatment_plans):
    """EP-Learner (Curth & van der Schaar 2023): efficient influence function."""
    from sklearn.linear_model import LogisticRegression
    feature_cols = [c for c in X_train.columns if c not in treatment_plans]
    X_tr = X_train[feature_cols].values
    X_te = X_test[feature_cols].values
    y_tr = y_train.values

    t_np = np.argmax(X_train[treatment_plans].values, axis=1).astype(int)
    prop_model = LogisticRegression(max_iter=500, random_state=SEED)
    prop_model.fit(X_tr, t_np)
    prop_proba = prop_model.predict_proba(X_tr)

    mu_models = {}
    for k, tp in enumerate(treatment_plans):
        mask = X_train[tp].values == 1
        if mask.sum() < 5:
            mu_models[tp] = None
            continue
        m = GradientBoostingRegressor(n_estimators=100, random_state=SEED)
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
        final = GradientBoostingRegressor(n_estimators=100, random_state=SEED)
        final.fit(X_tr, eif_k)
        predicted_outcomes[tp] = final.predict(X_te)
    predicted_outcomes["REC_TP"] = predicted_outcomes.idxmin(axis=1)
    return predicted_outcomes


def recommend_bites(X_train, y_train, X_test, treatment_plans):
    """BITES: MMD-balanced representation network with arm-specific heads."""
    try:
        import torch.nn as nn
    except ImportError:
        return recommend_s_learner(X_train, y_train, X_test, treatment_plans)

    feature_cols = [c for c in X_train.columns if c not in treatment_plans]
    X_np   = X_train[feature_cols].values.astype(np.float32)
    y_np   = y_train.values.astype(np.float32)
    t_np   = np.argmax(X_train[treatment_plans].values, axis=1).astype(np.int64)
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

    def mmd_rbf(z1, z2, g=1.0):
        def k(a, b): return torch.exp(-g * torch.cdist(a, b) ** 2)
        return k(z1,z1).mean() - 2*k(z1,z2).mean() + k(z2,z2).mean()

    import torch
    model = BITES_Net()
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse   = nn.MSELoss()
    X_t, y_t, t_t = (torch.tensor(X_np), torch.tensor(y_np),
                     torch.tensor(t_np))
    for _ in range(60):
        opt.zero_grad()
        z, outs = model(X_t)
        loss = sum(mse(outs[k][t_t==k], y_t[t_t==k])
                   for k in range(n_arms) if (t_t==k).any())
        reps = [z[t_t==k] for k in range(n_arms) if (t_t==k).any()]
        mmd  = sum(mmd_rbf(reps[i], reps[j])
                   for i in range(len(reps)) for j in range(i+1, len(reps)))
        (loss + 0.01 * mmd).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        _, outs_te = model(torch.tensor(X_te_np))
    predicted_outcomes = pd.DataFrame(index=X_test.index)
    for i, tp in enumerate(treatment_plans):
        predicted_outcomes[tp] = outs_te[i].numpy()
    predicted_outcomes["REC_TP"] = predicted_outcomes.idxmin(axis=1)
    return predicted_outcomes

# ============================================================
# Method registry  -- identical structure to pipeline.py
# ============================================================

def _wrap_sklearn(model_factory):
    def recommend_fn(X_train, y_train, X_test, treatment_plans):
        m = model_factory()
        m.fit(X_train, y_train)
        return recommend_treatment(m, X_test, treatment_plans)
    return recommend_fn


def _make_tabpfn():
    return TabPFNRegressor(
        model_path=TABPFN_CKPT_PATH,
        device=_TABPFN_DEVICE,
        ignore_pretraining_limits=True,
        random_state=SEED,
    )


# Five seeds for the FDR ensemble
_FDR_SEEDS = [42, 123, 456, 789, 2024]
_FDR_MAX_FEATURES = 40


def _select_features_for_tabpfn(X_train, y_train, treatment_plans, max_non_tp=_FDR_MAX_FEATURES):
    """Select top non-TP features by F-score; always retain TP columns."""
    from sklearn.feature_selection import f_regression
    non_tp = [c for c in X_train.columns if c not in treatment_plans]
    tp_cols = [c for c in X_train.columns if c in treatment_plans]
    if len(non_tp) <= max_non_tp:
        return list(X_train.columns)
    f_scores, _ = f_regression(X_train[non_tp].fillna(0), y_train)
    top_idx = np.argsort(f_scores)[::-1][:max_non_tp]
    return [non_tp[i] for i in sorted(top_idx)] + tp_cols


def _tabpfn_recommend_fn(X_train, y_train, X_test, treatment_plans):
    """FDR: multi-seed ensemble of TabPFN-3 with feature selection."""
    selected_cols = _select_features_for_tabpfn(X_train, y_train, treatment_plans)
    X_tr = X_train[selected_cols]
    X_te = X_test[selected_cols]
    sum_preds = {tp: np.zeros(len(X_te)) for tp in treatment_plans}
    for seed in _FDR_SEEDS:
        m = TabPFNRegressor(
            model_path=TABPFN_CKPT_PATH,
            device=_TABPFN_DEVICE,
            ignore_pretraining_limits=True,
            random_state=seed,
        )
        m.fit(X_tr, y_train)
        for tp in treatment_plans:
            X_cf = X_te.copy()
            X_cf[treatment_plans] = 0
            X_cf[tp] = 1
            sum_preds[tp] += m.predict(X_cf)
    n = len(_FDR_SEEDS)
    predicted_outcomes = pd.DataFrame(index=X_test.index)
    for tp in treatment_plans:
        predicted_outcomes[tp] = sum_preds[tp] / n
    predicted_outcomes["REC_TP"] = predicted_outcomes.idxmin(axis=1)
    return predicted_outcomes


ALL_METHODS = {
    # Classical models
    "FDR": _tabpfn_recommend_fn,
    "CB":  _wrap_sklearn(lambda: CatBoostRegressor(verbose=0, random_state=SEED)),
    "NN":  _wrap_sklearn(lambda: MLPRegressor(random_state=SEED, max_iter=1000)),
    "RF":  _wrap_sklearn(lambda: RandomForestRegressor(random_state=SEED, n_estimators=100)),
    "SVR": _wrap_sklearn(lambda: SVR()),
    "XGB": _wrap_sklearn(lambda: XGBRegressor(random_state=SEED, verbosity=0)),
    "LR":  _wrap_sklearn(lambda: LinearRegression()),
    # B. Meta-Learners
    "S_L":          recommend_s_learner,
    "X_L":          recommend_x_learner,
    "DR_L":         recommend_dr_learner,
    "R_L":          recommend_r_learner,
    # C. Neural Causal
    "CF":           recommend_causal_forest,
    # D. Modern SOTA
    "CUTS":         recommend_cuts,
    "EP_L":         recommend_ep_learner,
    "BITES":        recommend_bites,
}


# ============================================================
# Main OOD analysis pipeline
# ============================================================

def run_analysis_pipeline(
    data_name,
    treatment_plans,
    outcome_col,
    remove_cols,
    base_path=os.getcwd(),
    seed=SEED,
    methods=None,
):
    if methods is None:
        methods = ALL_METHODS

    input_path  = os.path.join(base_path, "input",  data_name)
    output_path = os.path.join(base_path, "output", data_name)
    ensure_dir(output_path)

    train_file = os.path.join(input_path, f"{data_name}_train.csv")
    test_file  = os.path.join(input_path, f"{data_name}_test.csv")

    print(f"  Loading data ...")
    X_train, y_train = preprocess_data(load_dataset(train_file), outcome_col, remove_cols, treatment_plans)
    test_data         = load_dataset(test_file)
    X_test,  y_test  = preprocess_data(test_data, outcome_col, remove_cols, treatment_plans)
    print(f"  Train: {X_train.shape}  Test: {X_test.shape}")

    summary_stats       = {"recovery": [], "rcb": []}
    performance_metrics = {}

    for name, recommend_fn in methods.items():
        print(f"    [{name}] running ...")
        try:
            # Standard sklearn-style models expose .fit() directly via _wrap_sklearn;
            # causal methods receive (X_train, y_train, X_test, treatment_plans).
            # Both signatures are called via recommend_fn -- the wrapper handles fit.
            rec_df = recommend_fn(X_train, y_train, X_test, treatment_plans)

            # Performance: refit a plain predict on X_test for metric logging
            # (causal methods don't expose .predict directly, so metrics only for
            # standard models where the factory is sklearn-compatible)
            try:
                m_eval = methods[name].__closure__  # check if _wrap_sklearn closure
                if m_eval is not None:
                    m_tmp = [c.cell_contents for c in m_eval
                             if callable(getattr(c.cell_contents, "fit", None))][0]()
                    m_tmp.fit(X_train, y_train)
                    mse, mae, r2, _ = evaluate_model(m_tmp, X_test, y_test)
                    performance_metrics[name] = {"Model": name, "MSE": mse, "MAE": mae, "R2": r2}
            except Exception:
                pass  # causal methods: skip per-model metrics

            combined_df, _, _ = compare_recommendations(
                rec_df, test_data, outcome_col, treatment_plans
            )
            combined_df.to_csv(
                os.path.join(output_path, f"{data_name}_{name}_REC.csv"), index=False
            )
            save_summary_stats(summary_stats, name, combined_df, outcome_col)
            print(f"    [{name}] done.")

        except Exception as e:
            print(f"    [WARN] Method '{name}' failed: {e}")

 
    # Save performance metrics (standard models only)
    if performance_metrics:
        pd.DataFrame.from_dict(performance_metrics, orient="index").to_csv(
            os.path.join(output_path, f"{data_name}_model_performance.csv")
        )
        print(f"  Performance metrics saved -> output/{data_name}/{data_name}_model_performance.csv")

    # RCB summary
    result = {"output_dir": output_path}
    if summary_stats["recovery"]:
        recovery_df = pd.DataFrame(summary_stats["recovery"]).pivot(
            index="Method", columns="FOLLOW_REC", values=["Count", "Recovery"]
        )
        recovery_df.columns = [
            "NotFollowing_Count", "Following_Count",
            "NotFollowing_Recovery", "Following_Recovery",
        ]
        recovery_df = recovery_df.reset_index()
        recovery_df["Recovery_Ratio"] = (
            (recovery_df["Following_Recovery"] / recovery_df["Following_Count"]) /
            (recovery_df["NotFollowing_Recovery"] / recovery_df["NotFollowing_Count"])
        )
        recovery_df.to_csv(
            os.path.join(output_path, f"{data_name}_Recovery_Ratio.csv"), index=False
        )
        result["recovery_summary"] = recovery_df

    if summary_stats["rcb"]:
        rcb_df = pd.DataFrame(summary_stats["rcb"]).pivot(
            index="Method", columns="FOLLOW_REC", values=["Count", "Avg_RCB_Score"]
        )
        rcb_df.columns = [
            "NotFollowing_Count", "Following_Count",
            "NotFollowing_Avg_RCB", "Following_Avg_RCB",
        ]
        rcb_df = rcb_df.reset_index()
        rcb_df.to_csv(
            os.path.join(output_path, f"{data_name}_RCB_Score_Comparison.csv"), index=False
        )
        result["rcb_summary"] = rcb_df

    return result


# ============================================================
# Batch runner
# ============================================================

def run_multiple_analyses():
    datasets = [

        {
            "data_name":       "OOD_multi_Trans_ART",
            "treatment_plans": ["TP1", "TP2", "TP3", "TP4"],
        },
    ]

    outcome_col = "RCB.score"
    remove_cols = [
        "Trial.ID", "resp.Chemosensitive", "resp.Chemoresistant", "resp.pCR",
        "RCB.category",
        "Chemo.NumCycles", "Chemo.first.Taxane", "Chemo.first.Anthracycline",
        "Chemo.second.Taxane", "Chemo.second.Anthracycline",
        "Chemo.any.Anthracycline", "Chemo.any.antiHER2",
    ]

    for cfg in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {cfg['data_name']}  |  Methods: {list(ALL_METHODS.keys())}")
        print(f"{'='*60}")
        run_analysis_pipeline(
            data_name=cfg["data_name"],
            treatment_plans=cfg["treatment_plans"],
            outcome_col=outcome_col,
            remove_cols=remove_cols,
        )


# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    run_multiple_analyses()