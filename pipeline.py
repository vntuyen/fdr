import os
import warnings
import numpy as np
import pandas as pd
import torch

# ============================================================
# Block ALL HuggingFace / telemetry network calls BEFORE
# importing tabpfn — must be set before the tabpfn import.
# ============================================================
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TABPFN_NO_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from sklearn.model_selection import KFold, StratifiedKFold
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

# ── Preprocessing integration ────────────────────────────────────────────────
from preprocessing import (preprocess_df_for_model, should_preprocess,
                           balance_training_arms, get_balance_config, should_balance)

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
            "Download it to that path before running (no network access allowed)."
        )
    if not os.access(TABPFN_CKPT_PATH, os.R_OK):
        raise PermissionError(f"TabPFN checkpoint is not readable: '{TABPFN_CKPT_PATH}'")
    size_mb = os.path.getsize(TABPFN_CKPT_PATH) / 1024 / 1024
    print(f"  [INFO] TabPFN checkpoint OK: '{TABPFN_CKPT_PATH}' ({size_mb:.0f} MB)")


def _check_cuda():
    if not torch.cuda.is_available():
        print("  [INFO] No GPU detected, using CPU.")
        return "cpu"

    cap  = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    print(f"  [INFO] GPU: {name}, compute capability: sm_{cap[0]}{cap[1]}")

    ver_parts = torch.__version__.split("+")[0].split(".")
    torch_major, torch_minor = int(ver_parts[0]), int(ver_parts[1])

    if cap == (7, 0) and (torch_major, torch_minor) >= (2, 6):
        print(
            f"  [WARN] sm_70 (V100) support was dropped in torch 2.6+. "
            f"You have torch {torch.__version__}. Falling back to CPU. "
            "Reinstall torch 2.5.x+cu121 to use this GPU."
        )
        return "cpu"

    print(f"  [INFO] CUDA sm_{cap[0]}{cap[1]} supported by torch {torch.__version__} — GPU enabled.")
    return "cuda"


_check_ckpt()

# ============================================================
# TabPFN factory
# ============================================================

_TABPFN_DEVICE = _check_cuda()
os.environ["TABPFN_DEVICE"] = _TABPFN_DEVICE


def _make_tabpfn() -> TabPFNRegressor:
    return TabPFNRegressor(
        model_path=TABPFN_CKPT_PATH,
        device=_TABPFN_DEVICE,
        ignore_pretraining_limits=True,
        random_state=SEED,
    )


# ============================================================
# Utility Functions
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_dataset(path: str, encoding: str = "ISO-8859-1") -> pd.DataFrame:
    return pd.read_csv(path, encoding=encoding, engine='python')


def preprocess_data(
    dataset: pd.DataFrame,
    outcome_col: str,
    remove_cols: list,
    treatment_plans: list = None,
) -> tuple:
    """
    Global preprocessing: impute + scale non-TP features.
    Treatment-plan indicator columns stay as binary 0/1 (counterfactual safety).
    Called ONCE per dataset before the fold loop.
    """
    drop = [c for c in remove_cols + [outcome_col] if c in dataset.columns]
    X    = dataset.drop(columns=drop)
    y    = dataset[outcome_col].astype(float)

    valid_idx = y.notna()
    X, y = X.loc[valid_idx], y[valid_idx]
    X = X.dropna(axis=1, how='all')

    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = pd.factorize(X[col])[0]

    tp_cols_in_X = [tp for tp in (treatment_plans or []) if tp in X.columns]
    X_tp    = X[tp_cols_in_X].copy()
    X_other = X.drop(columns=tp_cols_in_X)

    imputer = SimpleImputer(strategy='mean')
    X_other_imp = imputer.fit_transform(X_other)

    scaler = StandardScaler()
    X_other_scaled = scaler.fit_transform(X_other_imp)

    X_other_df = pd.DataFrame(X_other_scaled, columns=X_other.columns, index=X.index)
    X_final    = pd.concat([X_other_df, X_tp], axis=1)

    return X_final, y


# ============================================================
# Recommendation helpers
# ============================================================

def recommend_treatment(model, X_test: pd.DataFrame, treatment_plans: list) -> pd.DataFrame:
    """
    Generate counterfactual predictions for each treatment arm.
    Recommends the arm with the lowest predicted outcome (lower RCB = better).
    """
    predicted_outcomes = pd.DataFrame(index=X_test.index)

    for tp in treatment_plans:
        X_cf = X_test.copy()
        X_cf[treatment_plans] = 0
        X_cf[tp] = 1
        predicted_outcomes[tp] = model.predict(X_cf)

    predicted_outcomes['REC_TP'] = predicted_outcomes.idxmin(axis=1)
    return predicted_outcomes


def compare_recommendations(
    recommended_df: pd.DataFrame,
    original_df:    pd.DataFrame,
    outcome_col:    str,
    tp_cols:        list,
):
    recommended_df = recommended_df.copy()
    recommended_df['CURRENT_TP'] = original_df[tp_cols].idxmax(axis=1).values
    recommended_df['FOLLOW_REC'] = recommended_df['REC_TP'] == recommended_df['CURRENT_TP']
    combined = pd.concat(
        [original_df.reset_index(drop=True), recommended_df.reset_index(drop=True)],
        axis=1,
    )
    return combined, None, None




# ============================================================
# B. Meta-Learner Baselines
# ============================================================

def recommend_s_learner(X_train, y_train, X_test, treatment_plans):
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
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
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X_train.loc[mask, feature_cols], y_train[mask])
        predicted_outcomes[tp] = model.predict(X_test[feature_cols])

    predicted_outcomes['REC_TP'] = predicted_outcomes.idxmin(axis=1)
    return predicted_outcomes


def recommend_x_learner(X_train, y_train, X_test, treatment_plans):
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
        m = GradientBoostingRegressor(n_estimators=100, random_state=42)
        m.fit(X_tr[mask], y_train.values[mask])
        mu_models[tp] = m

    cate_models = {}
    for tp in treatment_plans:
        mask = arm_masks[tp]
        if mu_models[tp] is None or mask.sum() < 5:
            cate_models[tp] = None
            continue
        pseudo = y_train.values[mask] - np.mean(
            [mu_models[other].predict(X_tr[mask])
             for other, m in mu_models.items()
             if other != tp and m is not None],
            axis=0,
        ) if any(m is not None for k, m in mu_models.items() if k != tp) else np.zeros(mask.sum())

        cate_m = Ridge(alpha=1.0)
        cate_m.fit(X_tr[mask], pseudo)
        cate_models[tp] = cate_m

    predicted_outcomes = pd.DataFrame(index=X_test.index)
    baseline = np.mean([m.predict(X_te) for m in mu_models.values() if m is not None], axis=0)

    for tp in treatment_plans:
        if cate_models.get(tp) is not None:
            predicted_outcomes[tp] = baseline + cate_models[tp].predict(X_te)
        elif mu_models.get(tp) is not None:
            predicted_outcomes[tp] = mu_models[tp].predict(X_te)
        else:
            predicted_outcomes[tp] = float(y_train.mean())

    predicted_outcomes['REC_TP'] = predicted_outcomes.idxmin(axis=1)
    return predicted_outcomes


def _fit_propensity(X_train, treatment_plans):
    from sklearn.linear_model import LogisticRegression
    arm_labels = treatment_plans
    t_label = np.array([
        arm_labels.index(tp)
        for tp in treatment_plans
        for _ in range((X_train[tp] == 1).sum())
    ])
    X_reordered = np.vstack([
        X_train[[c for c in X_train.columns if c not in treatment_plans]].values[X_train[tp] == 1]
        for tp in treatment_plans
    ])
    prop = LogisticRegression(max_iter=500, random_state=42)
    prop.fit(X_reordered, t_label)
    return prop, arm_labels


def recommend_dr_learner(X_train, y_train, X_test, treatment_plans):
    feature_cols = [c for c in X_train.columns if c not in treatment_plans]
    X_tr = X_train[feature_cols].values
    X_te = X_test[feature_cols].values
    y_tr = y_train.values

    outcome_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    outcome_model.fit(X_train.values, y_tr)

    prop_model, _ = _fit_propensity(X_train, treatment_plans)
    prop_proba    = prop_model.predict_proba(X_tr)

    predicted_outcomes = pd.DataFrame(index=X_test.index)

    for i, tp in enumerate(treatment_plans):
        mask = X_train[tp].values == 1
        pi   = np.clip(prop_proba[:, i], 1e-6, 1 - 1e-6)

        X_cf = X_train.copy()
        X_cf[treatment_plans] = 0
        X_cf[tp] = 1
        mu_tp = outcome_model.predict(X_cf.values)

        dr_pseudo = mu_tp + mask.astype(float) * (y_tr - mu_tp) / pi
        cate_m = Ridge(alpha=1.0)
        cate_m.fit(X_tr, dr_pseudo)
        predicted_outcomes[tp] = cate_m.predict(X_te)

    predicted_outcomes['REC_TP'] = predicted_outcomes.idxmin(axis=1)
    return predicted_outcomes


def recommend_r_learner(X_train, y_train, X_test, treatment_plans):
    from sklearn.linear_model import LogisticRegression
    feature_cols = [c for c in X_train.columns if c not in treatment_plans]
    X_tr = X_train[feature_cols].values
    X_te = X_test[feature_cols].values
    y_tr = y_train.values

    m_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    m_model.fit(X_tr, y_tr)
    y_resid    = y_tr - m_model.predict(X_tr)
    baseline_te = m_model.predict(X_te)

    predicted_outcomes = pd.DataFrame(index=X_test.index)

    for tp in treatment_plans:
        t = X_train[tp].values.astype(float)
        e_model = LogisticRegression(max_iter=500, random_state=42)
        e_model.fit(X_tr, t.astype(int))
        e_hat    = np.clip(e_model.predict_proba(X_tr)[:, 1], 1e-6, 1 - 1e-6)
        t_resid  = t - e_hat
        safe_mask = np.abs(t_resid) > 1e-4

        if safe_mask.sum() < 5:
            predicted_outcomes[tp] = baseline_te
            continue

        tau_model = Ridge(alpha=1.0)
        tau_model.fit(X_tr[safe_mask],
                      y_resid[safe_mask] / t_resid[safe_mask],
                      sample_weight=t_resid[safe_mask] ** 2)
        predicted_outcomes[tp] = baseline_te + tau_model.predict(X_te)

    predicted_outcomes['REC_TP'] = predicted_outcomes.idxmin(axis=1)
    return predicted_outcomes


# ============================================================
# C. Causal Baselines
# ============================================================

def _build_tarnet_or_dragonnet(input_dim, n_arms, dragonnet=False):
    try:
        import torch.nn as nn
    except ImportError:
        return None

    class TARNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(input_dim, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
            )
            self.heads = nn.ModuleList([
                nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
                for _ in range(n_arms)
            ])
            if dragonnet:
                self.prop_head = nn.Sequential(
                    nn.Linear(64, 32), nn.ReLU(),
                    nn.Linear(32, n_arms), nn.Softmax(dim=1),
                )

        def forward(self, x, return_prop=False):
            z    = self.shared(x)
            outs = [h(z).squeeze(1) for h in self.heads]
            if dragonnet and return_prop:
                return outs, self.prop_head(z)
            return outs

    return TARNet()


def _train_tarnet(X_train, y_train, treatment_plans, dragonnet=False, epochs=50, lr=1e-3):
    try:
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError:
        return None, None

    feature_cols = [c for c in X_train.columns if c not in treatment_plans]
    X_np = X_train[feature_cols].values.astype(np.float32)
    y_np = y_train.values.astype(np.float32)
    t_np = np.argmax(X_train[treatment_plans].values, axis=1).astype(np.int64)

    model = _build_tarnet_or_dragonnet(X_np.shape[1], len(treatment_plans), dragonnet)
    if model is None:
        return None, None

    X_t, y_t, t_t = torch.tensor(X_np), torch.tensor(y_np), torch.tensor(t_np)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    mse = torch.nn.MSELoss()
    ce  = torch.nn.CrossEntropyLoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t, y_t, t_t), batch_size=64, shuffle=True
    )

    model.train()
    for _ in range(epochs):
        for xb, yb, tb in loader:
            optimiser.zero_grad()
            if dragonnet:
                outs, prop = model(xb, return_prop=True)
                loss = (
                    sum(mse(outs[k][tb == k], yb[tb == k])
                        for k in range(len(treatment_plans)) if (tb == k).any())
                    + 0.1 * ce(prop, tb)
                )
            else:
                outs = model(xb)
                loss = sum(
                    mse(outs[k][tb == k], yb[tb == k])
                    for k in range(len(treatment_plans)) if (tb == k).any()
                )
            if isinstance(loss, int):
                continue
            loss.backward()
            optimiser.step()

    model.eval()
    return model, feature_cols


def recommend_tarnet(X_train, y_train, X_test, treatment_plans):
    model, feature_cols = _train_tarnet(X_train, y_train, treatment_plans, dragonnet=False)
    if model is None:
        return recommend_t_learner_gb(X_train, y_train, X_test, treatment_plans)
    X_te_t = torch.tensor(X_test[feature_cols].values.astype(np.float32))
    with torch.no_grad():
        outs = model(X_te_t)
    predicted_outcomes = pd.DataFrame(index=X_test.index)
    for i, tp in enumerate(treatment_plans):
        predicted_outcomes[tp] = outs[i].numpy()
    predicted_outcomes['REC_TP'] = predicted_outcomes.idxmin(axis=1)
    return predicted_outcomes


def recommend_dragonnet(X_train, y_train, X_test, treatment_plans):
    model, feature_cols = _train_tarnet(X_train, y_train, treatment_plans, dragonnet=True)
    if model is None:
        return recommend_t_learner_gb(X_train, y_train, X_test, treatment_plans)
    X_te_t = torch.tensor(X_test[feature_cols].values.astype(np.float32))
    with torch.no_grad():
        outs = model(X_te_t)
    predicted_outcomes = pd.DataFrame(index=X_test.index)
    for i, tp in enumerate(treatment_plans):
        predicted_outcomes[tp] = outs[i].numpy()
    predicted_outcomes['REC_TP'] = predicted_outcomes.idxmin(axis=1)
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
    baseline_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    baseline_model.fit(X_tr, y_tr)
    baseline_te = baseline_model.predict(X_te)

    for tp in treatment_plans:
        t = X_train[tp].values.astype(float)
        if t.sum() < 5 or (1 - t).sum() < 5:
            predicted_outcomes[tp] = baseline_te
            continue
        try:
            cf = CausalForestDML(n_estimators=100, random_state=42, discrete_treatment=False)
            cf.fit(y_tr, t, X=X_tr, W=None)
            predicted_outcomes[tp] = baseline_te + cf.effect(X_te)
        except Exception:
            predicted_outcomes[tp] = baseline_te

    predicted_outcomes['REC_TP'] = predicted_outcomes.idxmin(axis=1)
    return predicted_outcomes


# ============================================================
# D. Modern SOTA Baselines
# ============================================================

def recommend_bites(X_train, y_train, X_test, treatment_plans):
    try:
        import torch.nn as nn
    except ImportError:
        return recommend_t_learner_gb(X_train, y_train, X_test, treatment_plans)

    feature_cols = [c for c in X_train.columns if c not in treatment_plans]
    X_np    = X_train[feature_cols].values.astype(np.float32)
    y_np    = y_train.values.astype(np.float32)
    t_np    = np.argmax(X_train[treatment_plans].values, axis=1).astype(np.int64)
    X_te_np = X_test[feature_cols].values.astype(np.float32)
    n_arms, in_dim = len(treatment_plans), X_np.shape[1]

    class BITES_Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.repr  = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(),
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
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse   = nn.MSELoss()
    X_t, y_t, t_t = torch.tensor(X_np), torch.tensor(y_np), torch.tensor(t_np)

    for _ in range(60):
        opt.zero_grad()
        z, outs = model(X_t)
        outcome_loss = sum(
            mse(outs[k][t_t == k], y_t[t_t == k])
            for k in range(n_arms) if (t_t == k).any()
        )
        arm_reprs = [z[t_t == k] for k in range(n_arms) if (t_t == k).any()]
        mmd_loss  = sum(
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
    predicted_outcomes['REC_TP'] = predicted_outcomes.idxmin(axis=1)
    return predicted_outcomes


def recommend_cuts(X_train, y_train, X_test, treatment_plans):
    feature_cols = [c for c in X_train.columns if c not in treatment_plans]
    X_tr = X_train[feature_cols].values
    X_te = X_test[feature_cols].values
    y_tr = y_train.values

    predicted_outcomes = pd.DataFrame(index=X_test.index)
    uncertainty        = pd.DataFrame(index=X_test.index)

    for tp in treatment_plans:
        mask = X_train[tp].values == 1
        if mask.sum() < 5:
            predicted_outcomes[tp] = y_tr.mean()
            uncertainty[tp]        = y_tr.std()
            continue
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X_tr[mask], y_tr[mask])
        residuals = np.abs(y_tr[mask] - model.predict(X_tr[mask]))
        predicted_outcomes[tp] = model.predict(X_te)
        uncertainty[tp]        = np.quantile(residuals, 0.9)

    ucb = predicted_outcomes[treatment_plans] + uncertainty[treatment_plans]
    predicted_outcomes['REC_TP'] = ucb.idxmin(axis=1)
    return predicted_outcomes


def recommend_ep_learner(X_train, y_train, X_test, treatment_plans):
    from sklearn.linear_model import LogisticRegression
    feature_cols = [c for c in X_train.columns if c not in treatment_plans]
    X_tr = X_train[feature_cols].values
    X_te = X_test[feature_cols].values
    y_tr = y_train.values

    t_np = np.argmax(X_train[treatment_plans].values, axis=1).astype(int)
    prop_model = LogisticRegression(max_iter=500, random_state=42)
    prop_model.fit(X_tr, t_np)
    prop_proba = prop_model.predict_proba(X_tr)

    mu_models = {}
    for k, tp in enumerate(treatment_plans):
        mask = X_train[tp].values == 1
        if mask.sum() < 5:
            mu_models[tp] = None
            continue
        m = GradientBoostingRegressor(n_estimators=100, random_state=42)
        m.fit(X_tr[mask], y_tr[mask])
        mu_models[tp] = m

    predicted_outcomes = pd.DataFrame(index=X_test.index)

    for k, tp in enumerate(treatment_plans):
        if mu_models[tp] is None:
            predicted_outcomes[tp] = y_tr.mean()
            continue
        mu_k  = mu_models[tp].predict(X_tr)
        e_k   = np.clip(prop_proba[:, k], 1e-6, 1.0)
        ind_k = (X_train[tp].values == 1).astype(float)
        eif_k = mu_k + (ind_k / e_k) * (y_tr - mu_k)

        final_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        final_model.fit(X_tr, eif_k)
        predicted_outcomes[tp] = final_model.predict(X_te)

    predicted_outcomes['REC_TP'] = predicted_outcomes.idxmin(axis=1)
    return predicted_outcomes


# ============================================================
# Unified Model/Method Registry
# ============================================================

def _wrap_sklearn(model_factory):
    """Wrap a sklearn-compatible model: fit → recommend_treatment."""
    def recommend_fn(X_train, y_train, X_test, treatment_plans):
        m = model_factory()
        m.fit(X_train, y_train)
        return recommend_treatment(m, X_test, treatment_plans)
    return recommend_fn


# Minimum training-arm samples required for TabPFN to fit a reliable arm model.
# Arms below this threshold fall back to the overall training-set mean.
_TABPFN_MIN_ARM_SAMPLES = 5


def _tabpfn_tlearner_fn(X_train, y_train, X_test, treatment_plans):
    """
    FDR (TabPFN) implemented as a T-Learner: one TabPFN per treatment arm.

    -----------------------------------------
      • Ridge regression: toggling TP2→TP1 applies a fixed linear shift
        (β_TP1 − β_TP2) regardless of which patients are in training.
        The prediction changes smoothly and predictably.

      • TabPFN (attention-based in-context learner): when asked to predict
        a TP2 patient with TP2=0, TP1=1, it receives an input pattern it
        has NEVER seen in training (all TP2 patients had TP2=1).  The
        attention mechanism has no analogous context to retrieve, producing
        erratic, unreliable counterfactual estimates → negative CAU.

    The T-Learner approach eliminates the OOD issue entirely:
      • Each arm's model is trained only on in-distribution data.
      • At inference, each model predicts the outcome for ALL test patients
        from their features alone — no TP column toggling, no OOD input.
      • Arms with < _TABPFN_MIN_ARM_SAMPLES training samples (e.g. TP3 n=3
        in multi_ARTemis) fall back to the training-set mean, which is a
        safe neutral estimate.
      • No arm balancing needed: each model sees only its own arm's data,
        so majority-arm dominance is not a factor.
    """
    feature_cols = [c for c in X_train.columns if c not in treatment_plans]
    predicted_outcomes = pd.DataFrame(index=X_test.index)

    for tp in treatment_plans:
        mask  = X_train[tp] == 1
        n_arm = int(mask.sum())
        if n_arm < _TABPFN_MIN_ARM_SAMPLES:
            predicted_outcomes[tp] = float(y_train.mean())
            print(f"      [FDR] {tp}: {n_arm} train samples < {_TABPFN_MIN_ARM_SAMPLES} → mean fallback")
            continue
        m = _make_tabpfn()
        m.fit(
            X_train.loc[mask, feature_cols].values,
            y_train[mask].values,
        )
        predicted_outcomes[tp] = m.predict(X_test[feature_cols].values)

    predicted_outcomes['REC_TP'] = predicted_outcomes.idxmin(axis=1)
    return predicted_outcomes

# ============================================================
# A. Classical Baselines
# ============================================================


STANDARD_MODELS = {
    "FDR": _tabpfn_tlearner_fn,   # T-Learner: arm-specific TabPFN (eliminates counterfactual OOD)
    "CB":  _wrap_sklearn(lambda: CatBoostRegressor(verbose=0)),
    "NN":  _wrap_sklearn(lambda: MLPRegressor(random_state=42, max_iter=1000)),
    "RF":  _wrap_sklearn(lambda: RandomForestRegressor(random_state=42, n_estimators=100)),
    "SVR": _wrap_sklearn(lambda: SVR()),
    "XGB": _wrap_sklearn(lambda: XGBRegressor(random_state=42, verbosity=0)),
    "LR":  _wrap_sklearn(lambda: LinearRegression()),
}

BASELINE_METHODS = {
    "CUTS":         recommend_cuts,
    "EP_L":         recommend_ep_learner,
    "DR_L":         recommend_dr_learner,
    "R_L":          recommend_r_learner,
    "BITES":        recommend_bites,
    "S_L":          recommend_s_learner,
    "X_L":          recommend_x_learner,
    "CF":           recommend_causal_forest,
}
    
METHOD_META = {
    "FDR":          ("FDR",            "Proposed"),
    "CB":           ("CatBoost",       "A. Classical"),
    "NN":           ("Neural Net",     "A. Classical"),
    "RF":           ("Random Forest",  "A. Classical"),
    "SVR":          ("SVR",            "A. Classical"),
    "XGB":          ("XGBoost",        "A. Classical"),
    "S_L":          ("S-Learner",      "B. Meta-Learner"),
    "X_L":          ("X-Learner",      "B. Meta-Learner"),
    "DR_L":         ("DR-Learner",     "B. Meta-Learner"),
    "R_L":          ("R-Learner",      "B. Meta-Learner"),
    "CF":           ("Causal Forest",  "C. Causal"),
    "CUTS":         ("CUTS",           "D. Modern SOTA"),
    "EP_L":         ("EP-Learner",     "D. Modern SOTA"),
    "BITES":        ("BITES",          "D. Modern SOTA"),
}

ALL_METHODS = {**STANDARD_MODELS, **BASELINE_METHODS}


# ============================================================
# Cross-Validation Pipeline
# ============================================================

def make_recommendations_cv(
    methods:         dict,
    data_name:       str,
    treatment_plans: list,
    outcome_col:     str,
    remove_cols:     list,
    input_path:      str,
    output_path:     str,
    seed:            int = 42,
    k:               int = 5,
):
    input_file = os.path.join(input_path, f"{data_name}.csv")
    df         = load_dataset(input_file)
    X_full, y_full = preprocess_data(df, outcome_col, remove_cols, treatment_plans)

    tp_cols_present = [tp for tp in treatment_plans if tp in df.columns]
    if tp_cols_present:
        arm_labels = df.loc[y_full.index, tp_cols_present].values.argmax(axis=1)
    else:
        print("  [WARN] Treatment columns not found in df; falling back to unstratified split.")
        arm_labels = np.zeros(len(X_full), dtype=int)

    min_arm_count = np.bincount(arm_labels).min()
    if min_arm_count < k:
        small_arms = [tp for i, tp in enumerate(tp_cols_present)
                      if (arm_labels == i).sum() < k]
        print(
            f"  [WARN] Arms {small_arms} have < {k} members — cannot stratify. "
            "Falling back to unstratified KFold."
        )
        splitter   = KFold(n_splits=k, shuffle=True, random_state=seed)
        split_iter = splitter.split(X_full)
    else:
        splitter   = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        split_iter = splitter.split(X_full, arm_labels)

    all_rec_dfs = {name: [] for name in methods}

    for fold, (train_idx, test_idx) in enumerate(split_iter):
        train_dist = {tp: int((arm_labels[train_idx] == i).sum())
                      for i, tp in enumerate(tp_cols_present)}
        test_dist  = {tp: int((arm_labels[test_idx]  == i).sum())
                      for i, tp in enumerate(tp_cols_present)}
        print(f"  Fold {fold + 1}/{k}  |  train={train_dist}  test={test_dist}")

        X_train = X_full.iloc[train_idx]
        X_test  = X_full.iloc[test_idx]
        y_train = y_full.iloc[train_idx]

        for name, recommend_fn in methods.items():
            try:
                # ── Step 1: column selection (FDR / LR / SVR / NN only) ──
                # Drops redundant and noise-amplified columns using training
                # data only.  Tree-based models receive unchanged DataFrames.
                X_tr_m, X_te_m = preprocess_df_for_model(
                    model_name=name,
                    data_name=data_name,
                    X_train=X_train,
                    X_test=X_test,
                    treatment_plans=treatment_plans,
                    verbose=False,        # set True to log dropped columns
                )

                # ── Step 2: arm balancing (FDR / LR / SVR / NN only) ─────
                # Downsamples the majority arm so no arm exceeds
                # balance_max_ratio × median_arm_size training samples.
                # Key fix for multi_ARTemis where TP2=45 vs TP3=3 causes
                # majority-arm dominance → biased counterfactuals → negative
                # CAU.  This is the ONLY effective lever for TabPFN (FDR)
                # since TabPFN has no regularisation parameter.
                y_tr_m = y_train
                if should_balance(name):
                    balance_ratio = get_balance_config(data_name)
                    if balance_ratio is not None:
                        X_tr_m, y_tr_m = balance_training_arms(
                            X_train=X_tr_m,
                            y_train=y_train,
                            treatment_plans=treatment_plans,
                            max_ratio=balance_ratio,
                            seed=seed,
                        )
                # ─────────────────────────────────────────────────────────

                rec_df = recommend_fn(X_tr_m, y_tr_m, X_te_m, treatment_plans)
                combined_df, _, _ = compare_recommendations(
                    rec_df, df.iloc[test_idx], outcome_col, treatment_plans
                )
                all_rec_dfs[name].append(combined_df)

            except Exception as e:
                print(f"    [WARN] Method '{name}' failed on fold {fold + 1}: {e}")

    final_dfs = {}
    for name, df_list in all_rec_dfs.items():
        if not df_list:
            print(f"  [SKIP] No results for method '{name}'")
            continue
        final_df = pd.concat(df_list, ignore_index=True)
        out_file = os.path.join(output_path, f"{data_name}_{name}_REC.csv")
        final_df.to_csv(out_file, index=False)
        final_dfs[name] = final_df

    return final_dfs


# ============================================================
# Entry Points
# ============================================================

def run_crossval_pipeline_entry(
    data_name,
    treatment_plans,
    outcome_col,
    remove_cols,
    base_path: str = os.getcwd(),
    seed:      int = 42,
    methods:   dict = None,
    k:         int = 5,
):
    input_path  = os.path.join(base_path, 'input',  data_name)
    output_path = os.path.join(base_path, 'output', data_name)
    ensure_dir(output_path)

    if methods is None:
        methods = ALL_METHODS

    print(f"\n{'='*60}")
    print(f"Dataset: {data_name}  |  Methods: {list(methods.keys())}")
    print(f"Split: {k}-fold StratifiedKFold (by arm)  |  each patient tested exactly once")
    print(f"{'='*60}")

    return make_recommendations_cv(
        methods, data_name, treatment_plans, outcome_col,
        remove_cols, input_path, output_path, seed=seed, k=k,
    )


def run_all_datasets(methods: dict = None, k: int = 5):
    datasets = [
        {
            "data_name": "GSE22226",
            "treatment_plans": ["AC_only", "AC_T", "AC_T_Herceptin", "AC_T_other"],
            "outcome_col": "RCB_category",
            "remove_cols": [
                "Group", "Accession", "Title",
                "Source_name_1", "Source_name_2", "Reference_Ch1", "treatment_plan",
                "Slide_name_Ch2", "I_spy_id_Ch2", "Experiment_name_Ch2",
                "Study_Ch2", "Tissue_Ch2",
                "Rcb_class_Ch2", "neoadjuvant_chemotherapy", "pcr",
                "relapse_free_survival_days", "relapse_free_survival_indicator",
                "overall_survival_days", "survival_status",
            ],
        },
        {
            "data_name": "GSE22358",
            "treatment_plans": ["Docetaxel_Cape", "Docetaxel_Cape_Tras"],
            "outcome_col": "RCB_category",
            "remove_cols": [
                "Group", "Accession", "Title",
                "Source_name_1", "Source_name_2", "Reference_Ch1",
                "Slidename_Ch2", "Sample_Ch2", "Study_Ch2", "Tissue_Ch2",
                "Response_Ch2", "Neoadjuvant_chemotherapy_Ch2",
            ],
        },

        {
            "data_name": "GSE41998",
            "treatment_plans": ["TP1", "TP2"],
            "outcome_col": "RCB.category",
            "remove_cols": [
                'Title', 'Source_name', 'Specimen_name',
                'Treatment_arm', 'Ac_response', 'Pcr', 'Pcrrcb1',
            ],
        },
        {
            "data_name": "clin_TransNEO",
            "treatment_plans": ["TP1", "TP2", "TP3", "TP4"],
            "outcome_col": "RCB.score",
            "remove_cols": [
                "Trial.ID", "resp.Chemosensitive", "resp.Chemoresistant",
                "resp.pCR", "RCB.category",
                "Chemo.NumCycles", "Chemo.first.Taxane", "Chemo.first.Anthracycline",
                "Chemo.second.Taxane", "Chemo.second.Anthracycline",
                "Chemo.any.Anthracycline", "Chemo.any.antiHER2",
            ],
        },
        {
            "data_name": "clin_ARTemis",
            "treatment_plans": ["TP1", "TP2", "TP3", "TP4"],
            "outcome_col": "RCB.score",
            "remove_cols": [
                "Trial.ID", "resp.Chemosensitive", "resp.Chemoresistant",
                "resp.pCR", "RCB.category",
                "Chemo.NumCycles", "Chemo.first.Taxane", "Chemo.first.Anthracycline",
                "Chemo.second.Taxane", "Chemo.second.Anthracycline",
                "Chemo.any.Anthracycline", "Chemo.any.antiHER2",
            ],
        },
        {
            "data_name": "multi_Trans_ART",
            "treatment_plans": ["TP1", "TP2", "TP3", "TP4"],
            "outcome_col": "RCB.score",
            "remove_cols": [
                "Trial.ID", "resp.Chemosensitive", "resp.Chemoresistant",
                "resp.pCR", "RCB.category",
                "Chemo.NumCycles", "Chemo.first.Taxane", "Chemo.first.Anthracycline",
                "Chemo.second.Taxane", "Chemo.second.Anthracycline",
                "Chemo.any.Anthracycline", "Chemo.any.antiHER2",
            ],
        },
        {
            "data_name": "multi_TransNEO",
            "treatment_plans": ["TP1", "TP2", "TP3", "TP4"],
            "outcome_col": "RCB.score",
            "remove_cols": [
                "Trial.ID", "resp.Chemosensitive", "resp.Chemoresistant",
                "resp.pCR", "RCB.score",
                "Chemo.NumCycles", "Chemo.first.Taxane", "Chemo.first.Anthracycline",
                "Chemo.second.Taxane", "Chemo.second.Anthracycline",
                "Chemo.any.Anthracycline", "Chemo.any.antiHER2",
            ],
        },
        {
            "data_name": "multi_ARTemis",
            "treatment_plans": ["TP1", "TP2", "TP3", "TP4"],
            "outcome_col": "RCB.score",
            "remove_cols": [
                "Trial.ID", "resp.Chemosensitive", "resp.Chemoresistant",
                "resp.pCR", "RCB.score",
                "Chemo.NumCycles", "Chemo.first.Taxane", "Chemo.first.Anthracycline",
                "Chemo.second.Taxane", "Chemo.second.Anthracycline",
                "Chemo.any.Anthracycline", "Chemo.any.antiHER2",
            ],
        },
    ]

    for cfg in datasets:
        print(f"\nRunning pipeline for dataset: {cfg['data_name']}")
        run_crossval_pipeline_entry(
            data_name=cfg['data_name'],
            treatment_plans=cfg['treatment_plans'],
            outcome_col=cfg['outcome_col'],
            remove_cols=cfg['remove_cols'],
            methods=methods,
            k=k,
        )


# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    run_all_datasets()