# -*- coding: utf-8 -*-
"""
preprocessing.py
================
Feature preprocessing and training-data balancing for linear/distance-based
models (FDR/TabPFN, LR, SVR, NN) across all pipeline datasets.

-----------------
A. Arm balancing  (fixes 1 + 2, works for ALL models including TabPFN)
   `balance_training_arms()` downsamples the majority arm so no arm is
   more than `balance_max_ratio × median_arm_size` of the training fold.
   Applied in the pipeline fold loop for FDR + linear models.

B. force_drop lists  (reduces noise-amplified features, helps all models)

DESIGN CONTRACT WITH pipeline.py
---------------------------------
preprocess_data() in the pipeline ALREADY applies StandardScaler globally.
ColumnSelector therefore operates in column-select-only mode (no re-scaling).
balance_training_arms() operates on pre-scaled DataFrames.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ============================================================
# Which models receive preprocessing
# ============================================================

# Scale/distance-sensitive models that benefit from column selection AND
# arm balancing.  Tree-based methods are immune to both and are excluded.
LINEAR_MODELS  = {"FDR", "LR", "SVR", "NN"}   # column selection
BALANCE_MODELS = {"LR", "SVR", "NN"}            # arm balancing (NOT FDR — T-Learner handles imbalance inherently)


def should_preprocess(model_name: str) -> bool:
    return model_name in LINEAR_MODELS


def should_balance(model_name: str) -> bool:
    return model_name in BALANCE_MODELS


# ============================================================
# Shared force_drop list for all multi-omics datasets
# ============================================================

_MULTI_OMICS_FORCE_DROP = [
    # Exact duplicates (r = 1.000)
    "STAT1.ssgsea.notnorm",
    "GGI.ssgsea.notnorm",
    "ESC.ssgsea.notnorm",
    "Chemo.first.Anthracycline",
    "Chemo.second.Taxane",
    # Near-perfect duplicates (r > 0.98)
    "Coding.TMB",
    "Chemo.any.antiHER2",
    "Danaher.Cytotoxic.cells",
    "TIDE.CD8",
    # Low-variance in raw space (std < 0.32) — noise-amplified after scaling
    "TIDE.TAM.M2",
    "TIDE.MDSC",
    "TIDE.CAF",
    "ESC.ssgsea.norm",
    "GGI.ssgsea.norm",
    "STAT1.ssgsea.norm",
    "GEP.ssgsea.norm",
    "CIN.Prop",
    "Histology",
]

# ============================================================
# Dataset-specific configs
# ============================================================
#
# Keys
# ----
# force_drop         list[str]   Always dropped (redundant/noisy columns).
# corr_threshold     float       Drop one of pair if |r| > this. Default 0.95.
# balance_max_ratio  float|None  Cap majority arm at this × median_arm_size.
#                                None = no balancing (default).
#                                Recommended: 3.0 for severely imbalanced sets.

DATASET_CONFIGS = {

    # ── multi_ARTemis ────────────────────────────────────────────────────
    # SEVERE imbalance: TP2=45, TP1=17, TP4=7, TP3=3.
    # Root cause of negative FDR CAU: Joint-model + TP-column toggling is
    # OOD for TabPFN — e.g. a TP2 patient with TP2=0, TP1=1 is a pattern
    # TabPFN has never seen in training → erratic counterfactual outputs.
    # Fix: FDR uses T-Learner (arm-specific models, no TP-toggling).
    #      T-Learner inherently handles imbalance so balance_max_ratio is
    #      NOT set for FDR.  LR/SVR/NN still use joint model and DO need
    #      balancing — the pipeline applies it via BALANCE_MODELS.
    "multi_ARTemis": {
        "corr_threshold":    0.95,
        "force_drop":        _MULTI_OMICS_FORCE_DROP,
        "balance_max_ratio": 3.0,   # applies to LR/SVR/NN (joint models only)
    },

    # ── multi_TransNEO ───────────────────────────────────────────────────
    # Moderate imbalance: TP1=26, TP2=75, TP3=30, TP4=16 (approx).
    # No balance needed; force_drop is the key fix here.
    "multi_TransNEO": {
        "corr_threshold": 0.95,
        "force_drop":     _MULTI_OMICS_FORCE_DROP,
    },

    # ── multi_Trans_ART (TransNEO + ARTemis combined) ────────────────────
    # Combined dataset; ARTemis imbalance is diluted by TransNEO.
    # Apply light balancing just in case.
    "multi_Trans_ART": {
        "corr_threshold":    0.95,
        "force_drop":        _MULTI_OMICS_FORCE_DROP,
        "balance_max_ratio": 4.0,
    },

    # ── OOD datasets ─────────────────────────────────────────────────────
    "OOD_multi_Trans_ART": {
        "corr_threshold":    0.95,
        "force_drop":        _MULTI_OMICS_FORCE_DROP,
        "balance_max_ratio": 4.0,
    },
    "OOD_multi_3TS_Trans_ART": {
        "corr_threshold":    0.95,
        "force_drop":        _MULTI_OMICS_FORCE_DROP,
        "balance_max_ratio": 4.0,
    },

    # ── Clinical-only datasets ────────────────────────────────────────────
    "clin_TransNEO":      {"corr_threshold": 0.95, "force_drop": []},
    "clin_ARTemis":       {"corr_threshold": 0.95, "force_drop": []},
    "GSE41998":      {"corr_threshold": 0.95, "force_drop": []},
    "GSE22226": {"corr_threshold": 0.95, "force_drop": []},
    "GSE22358":      {"corr_threshold": 0.95, "force_drop": []},
}

_DEFAULTS = {
    "corr_threshold":    0.95,
    "force_drop":        [],
    "force_keep":        [],
    "balance_max_ratio": None,
}


# ============================================================
# Arm balancing
# ============================================================

def balance_training_arms(
    X_train:         pd.DataFrame,
    y_train:         pd.Series,
    treatment_plans: list,
    max_ratio:       float = 3.0,
    seed:            int   = 42,
) -> tuple:
    """
    Downsample majority treatment arms so no arm exceeds
    `max(5, median_arm_size * max_ratio)` training samples.

    Addresses two failure modes:
    (a) Model overwhelmed by majority arm → biased counterfactuals.
    (b) Minority arm (e.g. TP3 n=3) too small to learn from reliably.

    Only downsamples — never synthesises data — so no leakage is introduced.
    Fit-on-train safe: the median is computed from the current training fold.

    Parameters
    ----------
    X_train          : pre-scaled DataFrame from the current fold
    y_train          : outcome Series aligned with X_train
    treatment_plans  : list of TP column names
    max_ratio        : cap = max(5, median_arm_size × max_ratio)
    seed             : numpy random seed for reproducibility

    Returns
    -------
    X_train_balanced, y_train_balanced
    """
    rng = np.random.RandomState(seed)

    # Arm sizes in the current training fold
    arm_sizes = {tp: int(X_train[tp].sum()) for tp in treatment_plans
                 if tp in X_train.columns}
    nonempty  = [s for s in arm_sizes.values() if s > 0]

    if not nonempty:
        return X_train, y_train

    median_sz = float(np.median(nonempty))
    cap       = max(5, int(median_sz * max_ratio))

    keep_idx = []
    for tp, size in arm_sizes.items():
        idxs = X_train.index[X_train[tp] == 1].tolist()
        if size > cap:
            idxs = rng.choice(idxs, cap, replace=False).tolist()
            print(f"      [Balance] {tp}: {size} → {cap} samples "
                  f"(cap = {max_ratio}× median {median_sz:.0f})")
        keep_idx.extend(idxs)

    keep_idx_sorted = sorted(keep_idx)
    return X_train.loc[keep_idx_sorted], y_train.loc[keep_idx_sorted]


def get_balance_config(data_name: str) -> float | None:
    """Return balance_max_ratio for the dataset, or None if not configured."""
    cfg = {**_DEFAULTS, **DATASET_CONFIGS.get(data_name, {})}
    return cfg.get("balance_max_ratio")


# ============================================================
# ColumnSelector — column dropping on pre-scaled DataFrames
# ============================================================

class ColumnSelector:
    """
    Determines which feature columns to drop, then filters DataFrames.
    Operates on already-scaled DataFrames (output of preprocess_data).
    Treatment-plan indicator columns are always preserved.
    """

    def __init__(self, data_name: str = "", model_name: str = "",
                 treatment_plans: list = None):
        cfg = {**_DEFAULTS, **DATASET_CONFIGS.get(data_name, {})}

        self.data_name        = data_name
        self.model_name       = model_name
        self.treatment_plans  = set(treatment_plans or [])
        self.corr_threshold   = cfg["corr_threshold"]
        self._force_drop      = set(cfg.get("force_drop", []))
        self._force_keep      = set(cfg.get("force_keep", [])) | self.treatment_plans

        self._drop_set     = None
        self.keep_cols_    = None
        self.drop_reasons_ = {}
        self.n_original_   = None
        self.n_kept_       = None

    def fit(self, X_train: pd.DataFrame) -> "ColumnSelector":
        all_cols         = X_train.columns.tolist()
        self.n_original_ = len(all_cols)
        drop_set         = set()

        # Step 1: explicit force_drop
        for col in self._force_drop:
            if col in X_train.columns and col not in self._force_keep:
                drop_set.add(col)
                self.drop_reasons_[col] = "force_drop (known redundant/noisy)"

        # Step 2: greedy correlation dedup (scale-invariant)
        surviving = [c for c in all_cols if c not in drop_set]
        if len(surviving) > 1:
            corr = X_train[surviving].corr().abs()
            for col in surviving:
                if col in drop_set or col in self._force_keep:
                    continue
                for other in surviving:
                    if other == col or other in drop_set or other in self._force_keep:
                        continue
                    if corr.loc[col, other] > self.corr_threshold:
                        drop_set.add(other)
                        if other not in self.drop_reasons_:
                            self.drop_reasons_[other] = (
                                f"correlated with '{col}' "
                                f"(|r|={corr.loc[col,other]:.3f} > {self.corr_threshold})"
                            )

        self._drop_set   = drop_set
        self.keep_cols_  = [c for c in all_cols if c not in drop_set]
        self.n_kept_     = len(self.keep_cols_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._drop_set is None:
            raise RuntimeError("Call fit() before transform().")
        cols = [c for c in self.keep_cols_ if c in X.columns]
        return X[cols]

    def fit_transform(self, X_train: pd.DataFrame,
                      X_test: pd.DataFrame):
        self.fit(X_train)
        return self.transform(X_train), self.transform(X_test)

    def log(self, verbose: bool = True):
        if not verbose:
            return
        n_dropped = self.n_original_ - self.n_kept_
        print(f"  [Preprocessor] {self.data_name}/{self.model_name}: "
              f"{self.n_original_} → {self.n_kept_} features  "
              f"(dropped {n_dropped}, corr_thresh={self.corr_threshold})")
        for col, reason in self.drop_reasons_.items():
            print(f"      DROP  {col}  [{reason}]")

    def summary(self) -> dict:
        return {
            "data_name":      self.data_name,
            "model_name":     self.model_name,
            "n_original":     self.n_original_,
            "n_kept":         self.n_kept_,
            "n_dropped":      (self.n_original_ - self.n_kept_) if self.n_kept_ else None,
            "corr_threshold": self.corr_threshold,
            "keep_cols":      self.keep_cols_,
            "drop_reasons":   self.drop_reasons_,
        }


# ============================================================
# Backward-compatible Preprocessor (raw-data mode, includes scaler)
# ============================================================

class Preprocessor:
    """Full preprocessor for raw (unscaled) data. Includes StandardScaler."""

    def __init__(self, data_name: str = "", model_name: str = ""):
        cfg = {**_DEFAULTS, **DATASET_CONFIGS.get(data_name, {})}
        self.data_name      = data_name
        self.model_name     = model_name
        self.corr_threshold = cfg["corr_threshold"]
        self.var_threshold  = cfg.get("var_threshold", 0.01)
        self.alpha_         = cfg.get("alpha", 10.0)
        self.force_drop     = set(cfg.get("force_drop", []))
        self.force_keep     = set(cfg.get("force_keep", []))
        self._scaler        = StandardScaler()
        self._keep_mask     = None
        self.keep_names_    = None
        self.n_dropped_     = None
        self.drop_reasons_  = {}

    def fit(self, X_train: np.ndarray, feature_names: list) -> "Preprocessor":
        df_tr = pd.DataFrame(X_train, columns=feature_names)
        stds  = df_tr.std()
        drop  = set(self.force_drop)

        for col in self.force_drop:
            if col in feature_names:
                self.drop_reasons_[col] = "force_drop"

        for col in feature_names:
            if col in drop or col in self.force_keep:
                continue
            if stds[col] < self.var_threshold:
                drop.add(col)
                self.drop_reasons_[col] = f"low_variance (std={stds[col]:.4f})"

        surviving = [c for c in feature_names if c not in drop]
        if len(surviving) > 1:
            corr = df_tr[surviving].corr().abs()
            for col in surviving:
                if col in drop or col in self.force_keep:
                    continue
                for other in surviving:
                    if other == col or other in drop or other in self.force_keep:
                        continue
                    if corr.loc[col, other] > self.corr_threshold:
                        drop.add(other)
                        if other not in self.drop_reasons_:
                            self.drop_reasons_[other] = (
                                f"correlated with '{col}' "
                                f"(|r|={corr.loc[col,other]:.3f} > {self.corr_threshold})")

        self._keep_mask  = np.array([c not in drop for c in feature_names])
        self.keep_names_ = [c for c in feature_names if c not in drop]
        self.n_dropped_  = int((~self._keep_mask).sum())
        self._scaler.fit(X_train[:, self._keep_mask])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._keep_mask is None:
            raise RuntimeError("Call fit() before transform().")
        return self._scaler.transform(X[:, self._keep_mask])

    def fit_transform(self, X_train: np.ndarray, X_test: np.ndarray,
                      feature_names: list):
        self.fit(X_train, feature_names)
        return self.transform(X_train), self.transform(X_test)

    def log(self, verbose: bool = True):
        if not verbose:
            return
        print(f"  [Preprocessor] {self.data_name}/{self.model_name}: "
              f"kept {len(self.keep_names_)}, dropped {self.n_dropped_}")

    def summary(self) -> dict:
        return {"data_name": self.data_name, "model_name": self.model_name,
                "n_kept": len(self.keep_names_) if self.keep_names_ else None,
                "n_dropped": self.n_dropped_, "alpha": self.alpha_,
                "drop_reasons": self.drop_reasons_}


# ============================================================
# Pipeline integration functions
# ============================================================

def preprocess_df_for_model(model_name:      str,
                             data_name:       str,
                             X_train:         pd.DataFrame,
                             X_test:          pd.DataFrame,
                             treatment_plans: list = None,
                             verbose:         bool = False):
    """
    Column-selection preprocessing for the pipeline fold loop.
    Called once per model per fold in make_recommendations_cv().
    Tree-based models receive unchanged DataFrames.
    """
    if not should_preprocess(model_name):
        return X_train, X_test

    selector = ColumnSelector(
        data_name=data_name,
        model_name=model_name,
        treatment_plans=treatment_plans,
    )
    X_train_out, X_test_out = selector.fit_transform(X_train, X_test)
    selector.log(verbose=verbose)
    return X_train_out, X_test_out


def preprocess_for_model(model_name: str, data_name: str,
                         X_train: np.ndarray, X_test: np.ndarray,
                         feature_names: list, verbose: bool = False):
    """Numpy-array version for standalone / unit-test use on raw unscaled data."""
    if not should_preprocess(model_name):
        return X_train, X_test, None
    cfg   = {**_DEFAULTS, **DATASET_CONFIGS.get(data_name, {})}
    alpha = cfg.get("alpha", 10.0)
    pre   = Preprocessor(data_name=data_name, model_name=model_name)
    X_tr, X_te = pre.fit_transform(X_train, X_test, feature_names)
    pre.log(verbose=verbose)
    return X_tr, X_te, alpha