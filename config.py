# -*- coding: utf-8 -*-
"""
config.py
=========
Single source of truth for:
  - Environment / runtime setup (HF/TabPFN offline flags, CUDA checks, SEED)
  - Dataset registry: per-dataset treatment plans, outcome column, columns to
    remove, file layout (CV vs OOD), and naming/plotting metadata

This module has NO knowledge of any specific method (FDR or baselines) and
NO knowledge of evaluation/plotting. It only prepares data and describes the
experimental scenarios. fdr.py / baselines.py / evaluation.py / 
run_experiments.py all import from here.
"""

import os
import warnings
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")

# ============================================================
# Environment setup (must happen before importing tabpfn)
# ============================================================

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TABPFN_NO_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


SEED = 42

TABPFN_CKPT_PATH = "/scratch/sq95/tv9849/FDR/tabpfn/tabpfn-v3-regressor-v3_default.ckpt"

# Ten seeds shared by the repeated-CV driver and the FDR OOD-ensemble so
# both pipelines' notion of "a repeat / ensemble member" stays consistent.

REPEAT_SEEDS = [42, 123, 456, 789, 2024, 7, 99, 1337, 31415, 271828]


def set_seed(seed: int):
    """Update the module-level SEED read by every model factory at call time."""
    global SEED
    SEED = seed


# ============================================================
# Startup checks: TabPFN checkpoint + CUDA
# ============================================================

def check_ckpt():
    if not os.path.isfile(TABPFN_CKPT_PATH):
        raise FileNotFoundError(
            f"TabPFN checkpoint not found: '{TABPFN_CKPT_PATH}'\n"
            "Download it to that path before running (no network access allowed)."
        )
    if not os.access(TABPFN_CKPT_PATH, os.R_OK):
        raise PermissionError(f"TabPFN checkpoint is not readable: '{TABPFN_CKPT_PATH}'")
    size_mb = os.path.getsize(TABPFN_CKPT_PATH) / 1024 / 1024
    print(f"  [INFO] TabPFN checkpoint OK: '{TABPFN_CKPT_PATH}' ({size_mb:.0f} MB)")


def check_cuda():
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
            f"  [WARN] sm_70 (V100) support was dropped in torch 2.6+. "
            f"You have torch {torch.__version__}. Falling back to CPU. "
            "Reinstall torch 2.5.x+cu121 to use this GPU."
        )
        return "cpu"

    print(f"  [INFO] CUDA sm_{cap[0]}{cap[1]} supported by torch {torch.__version__} — GPU enabled.")
    return "cuda"


# Run checks once at import time 
# behaviour: fail fast if the checkpoint is missing).
check_ckpt()
TABPFN_DEVICE = check_cuda()
os.environ["TABPFN_DEVICE"] = TABPFN_DEVICE


# ============================================================
# Generic I/O helpers
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_dataset(path: str, encoding: str = "ISO-8859-1") -> pd.DataFrame:
    return pd.read_csv(path, encoding=encoding, engine="python")


# ============================================================
# Dataset registry
# ============================================================
#
# Two SCENARIOS adapt FDR (and all baselines) to different evaluation
# settings -- these are not different methods, just different ways of
# splitting train/test for the same datasets:
#
#   "cv"  : repeated k-fold cross-validation, train and test drawn from the
#           same cohort (in-distribution).
#   "ood" : fixed train/test split where the test cohort is a different
#           study to the training cohort (out-of-distribution generalisation).
#
# `scenario` below records which evaluation protocol each dataset uses.
#
# Restricted to the 6 TransNEO/ARTemis-derived datasets (clinical and
# multi-omics views of the same two cohorts, plus their pooled-CV and OOD
# variants). Every dataset reported
# shares a consistent neoadjuvant-therapy setting -- same outcome
# (RCB.score), same treatment-plan structure (TP1-TP4), and a true
# same-patient clinical/multi-omics pairing for TransNEO and ARTemis. This
# keeps every comparison in evaluation.py (CAU/RRD/RR, the paired
# Clinical-vs-Multi-omics figure, multiplicity correction, etc.)
# apples-to-apples across datasets.

DATASET_REGISTRY = {
    "clin_TransNEO": {
        "scenario": "cv",
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
    "clin_ARTemis": {
        "scenario": "cv",
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
    "multi_Trans_ART": {
        "scenario": "cv",
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
    "multi_TransNEO": {
        "scenario": "cv",
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
    "multi_ARTemis": {
        "scenario": "cv",
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
    # ── OOD scenario: fixed train/test split, test cohort != train cohort ──
    "OOD_multi_Trans_ART": {
        "scenario": "ood",
        "treatment_plans": ["TP1", "TP2", "TP3", "TP4"],
        "outcome_col": "RCB.score",
        "remove_cols": [
            "Trial.ID", "resp.Chemosensitive", "resp.Chemoresistant", "resp.pCR",
            "RCB.category",
            "Chemo.NumCycles", "Chemo.first.Taxane", "Chemo.first.Anthracycline",
            "Chemo.second.Taxane", "Chemo.second.Anthracycline",
            "Chemo.any.Anthracycline", "Chemo.any.antiHER2",
        ],
    },
}

CV_DATASETS  = [d for d, c in DATASET_REGISTRY.items() if c["scenario"] == "cv"]
OOD_DATASETS = [d for d, c in DATASET_REGISTRY.items() if c["scenario"] == "ood"]

DATASET_NAME_MAP = {
    "clin_TransNEO":       "TransNEO clinical",
    "clin_ARTemis":        "ARTemis clinical",
    "multi_TransNEO":      "TransNEO multi-omics",
    "multi_ARTemis":       "ARTemis multi-omics",
    "multi_Trans_ART":     "Combined TransNEO + ARTemis multi-omics CV",
    "OOD_multi_Trans_ART": "TransNEO and ARTemis multi-omics OOD",
}

DATASET_TITLES = {
    "clin_ARTemis":        "ARTemis Clinical Dataset",
    "clin_TransNEO":       "TransNEO Clinical Dataset",
    "multi_ARTemis":       "ARTemis Multi-omics Dataset",
    "multi_TransNEO":      "TransNEO Multi-omics Dataset",
    "multi_Trans_ART":     "Multi-omics Datasets CV: TransNEO and ARTemis",
    "OOD_multi_Trans_ART": "Multi-omics Datasets OOD: TransNEO train, ARTemis test",
}

DATASET_ROW_ORDER = list(DATASET_NAME_MAP.values())


DATASET_MODALITY = {
    "clin_TransNEO":  "Clinical",
    "clin_ARTemis":   "Clinical",
    "multi_TransNEO": "Multi-omics",
    "multi_ARTemis":  "Multi-omics",
}


CLINICAL_MULTIOMICS_PAIRS = {
    "TransNEO": {"clinical": "clin_TransNEO", "multiomics": "multi_TransNEO"},
    "ARTemis":  {"clinical": "clin_ARTemis",  "multiomics": "multi_ARTemis"},
}


MULTIOMICS_DATASETS = [
    "multi_TransNEO", "multi_ARTemis", "multi_Trans_ART", "OOD_multi_Trans_ART",
]


# ============================================================
# Method registry metadata (names/colours/grouping for plots & tables)
# ============================================================
#
# 10 methods (FDR + 9 baselines)

METHOD_META = {
    "FDR":   ("FDR",           "Proposed"),
    "CB":    ("CatBoost",      "A. Classical"),
    "XGB":   ("XGBoost",       "A. Classical"),
    "S_L":   ("S-Learner",     "B. Meta-Learner"),
    "X_L":   ("X-Learner",     "B. Meta-Learner"),
    "DR_L":  ("DR-Learner",    "B. Meta-Learner"),
    "R_L":   ("R-Learner",     "B. Meta-Learner"),
    "CF":    ("Causal Forest", "C. Causal"),
    "CUTS":  ("CUTS",          "D. Modern SOTA"),
    "BITES": ("BITES",         "D. Modern SOTA"),
}

ALL_METHOD_COL_ORDER = [
    "FDR", "CB", "XGB",
    "S_L", "X_L", "DR_L", "R_L",
    "CF", "CUTS", "BITES",
]

MODELS_COLOUR = {
    "FDR":   "blue",
    "CB":    "cyan",
    "XGB":   "pink",
    "S_L":   "mediumseagreen",
    "X_L":   "steelblue",
    "DR_L":  "tomato",
    "R_L":   "darkorange",
    "CF":    "mediumpurple",
    "CUTS":  "saddlebrown",
    "BITES": "deeppink",
}


# ============================================================
# Global preprocessing: impute + scale (called once per dataset/split,
# BEFORE the fold loop or the fixed OOD train/test split)
# ============================================================

def preprocess_data(
    dataset: pd.DataFrame,
    outcome_col: str,
    remove_cols: list,
    treatment_plans: list = None,
) -> tuple:
    """
    Global preprocessing: impute + scale non-TP features.
    Treatment-plan indicator columns stay as binary 0/1 (counterfactual safety).
    """
    drop = [c for c in remove_cols + [outcome_col] if c in dataset.columns]
    X = dataset.drop(columns=drop)
    y = dataset[outcome_col].astype(float)

    valid_idx = y.notna()
    X, y = X.loc[valid_idx], y[valid_idx]
    X = X.dropna(axis=1, how="all")

    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = pd.factorize(X[col])[0]

    tp_cols_in_X = [tp for tp in (treatment_plans or []) if tp in X.columns]
    X_tp = X[tp_cols_in_X].copy()
    X_other = X.drop(columns=tp_cols_in_X)

    imputer = SimpleImputer(strategy="mean")
    X_other_imp = imputer.fit_transform(X_other)

    scaler = StandardScaler()
    X_other_scaled = scaler.fit_transform(X_other_imp)

    X_other_df = pd.DataFrame(X_other_scaled, columns=X_other.columns, index=X.index)
    X_final = pd.concat([X_other_df, X_tp], axis=1)

    return X_final, y


def make_cv_splitter(X_full, df, treatment_plans, k, seed):
    """
    Build a fold iterator, stratified by treatment arm when every arm has
    >= k members, falling back to plain KFold otherwise.
    Returns (split_iter, arm_labels, tp_cols_present).
    """
    tp_cols_present = [tp for tp in treatment_plans if tp in df.columns]
    if tp_cols_present:
        arm_labels = df.loc[X_full.index, tp_cols_present].values.argmax(axis=1)
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
        splitter = KFold(n_splits=k, shuffle=True, random_state=seed)
        split_iter = splitter.split(X_full)
    else:
        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        split_iter = splitter.split(X_full, arm_labels)

    return split_iter, arm_labels, tp_cols_present


# ============================================================
# Per-model column selection ("LINEAR_MODELS") + arm balancing
# ============================================================
#
# Scale/distance-sensitive models benefit from dropping redundant/noisy
# columns; tree-based and meta-learner methods are immune and are excluded.
# Now that Neural Net, SVR, and Linear Regression have been removed from
# the baseline set (see config.METHOD_META), FDR is the ONLY method that
# still needs this column-selection step. Arm balancing (downsampling the
# majority treatment arm) was only ever needed for those JOINT linear/
# distance models -- with them removed, BALANCE_MODELS is intentionally
# empty: FDR uses a T-Learner (one model per arm), which handles imbalance
# inherently and must NOT be balanced, since balancing would throw away
# minority-arm signal each arm model needs, and every remaining baseline
# (CatBoost, XGBoost, the meta-learners, Causal Forest, CUTS, EP-Learner,
# BITES) either is tree-based/ensemble-based (balancing-insensitive) or
# already handles arm imbalance internally via its own estimator (e.g.
# propensity weighting in the meta-learners).

LINEAR_MODELS = {"FDR"}    # column selection
BALANCE_MODELS = set()     # arm balancing -- intentionally empty, see above


def should_preprocess(model_name: str) -> bool:
    return model_name in LINEAR_MODELS


def should_balance(model_name: str) -> bool:
    return model_name in BALANCE_MODELS


# ---- Shared force_drop list for all multi-omics datasets ----

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

# Keys
# ----
# force_drop         list[str]   Always dropped (redundant/noisy columns).
# corr_threshold     float       Drop one of pair if |r| > this. Default 0.95.
# balance_max_ratio  float|None  Cap majority arm at this x median_arm_size.
#                                NOTE: BALANCE_MODELS is currently empty (the
#                                joint linear/distance baselines that needed
#                                arm balancing -- LR, SVR, NN -- were removed
#                                from the method set), so this setting is
#                                presently inert for every method, including
#                                FDR, which never balances regardless of this
#                                setting -- see should_balance(). Left in
#                                place as dataset-imbalance documentation and
#                                in case a future balancing-sensitive joint
#                                baseline is reintroduced.

PREPROCESS_CONFIGS = {
    # SEVERE imbalance: TP2=45, TP1=17, TP4=7, TP3=3. FDR uses a T-Learner
    # (arm-specific models, no TP-column toggling) so it inherently handles
    # this without balance_max_ratio.
    "multi_ARTemis": {
        "corr_threshold":    0.95,
        "force_drop":        _MULTI_OMICS_FORCE_DROP,
        "balance_max_ratio": 3.0,
    },
    # Moderate imbalance; force_drop is the key fix here, no balancing needed.
    "multi_TransNEO": {
        "corr_threshold": 0.95,
        "force_drop":     _MULTI_OMICS_FORCE_DROP,
    },
    # Combined dataset; ARTemis imbalance diluted by TransNEO. Light balance
    # just in case.
    "multi_Trans_ART": {
        "corr_threshold":    0.95,
        "force_drop":        _MULTI_OMICS_FORCE_DROP,
        "balance_max_ratio": 4.0,
    },
    # OOD scenario, same multi-omics feature set.
    "OOD_multi_Trans_ART": {
        "corr_threshold":    0.95,
        "force_drop":        _MULTI_OMICS_FORCE_DROP,
        "balance_max_ratio": 4.0,
    },
    # Clinical-only datasets: no engineered force_drop list needed.
    "clin_TransNEO": {"corr_threshold": 0.95, "force_drop": []},
    "clin_ARTemis":  {"corr_threshold": 0.95, "force_drop": []},
}

_PREPROCESS_DEFAULTS = {
    "corr_threshold":    0.95,
    "force_drop":        [],
    "force_keep":        [],
    "balance_max_ratio": None,
}


def get_balance_config(data_name: str):
    """Return balance_max_ratio for the dataset, or None if not configured."""
    cfg = {**_PREPROCESS_DEFAULTS, **PREPROCESS_CONFIGS.get(data_name, {})}
    return cfg.get("balance_max_ratio")


class ColumnSelector:
    """
    Determines which feature columns to drop, then filters DataFrames.
    Operates on already-scaled DataFrames (output of preprocess_data).
    Treatment-plan indicator columns are always preserved.

    Step 1: explicit force_drop (known redundant/noisy columns).
    Step 2: greedy correlation dedup (|r| > corr_threshold) on training data.
    """

    def __init__(self, data_name: str = "", model_name: str = "",
                 treatment_plans: list = None):
        cfg = {**_PREPROCESS_DEFAULTS, **PREPROCESS_CONFIGS.get(data_name, {})}

        self.data_name = data_name
        self.model_name = model_name
        self.treatment_plans = set(treatment_plans or [])
        self.corr_threshold = cfg["corr_threshold"]
        self._force_drop = set(cfg.get("force_drop", []))
        self._force_keep = set(cfg.get("force_keep", [])) | self.treatment_plans

        self._drop_set = None
        self.keep_cols_ = None
        self.drop_reasons_ = {}
        self.n_original_ = None
        self.n_kept_ = None

    def fit(self, X_train: pd.DataFrame) -> "ColumnSelector":
        all_cols = X_train.columns.tolist()
        self.n_original_ = len(all_cols)
        drop_set = set()

        for col in self._force_drop:
            if col in X_train.columns and col not in self._force_keep:
                drop_set.add(col)
                self.drop_reasons_[col] = "force_drop (known redundant/noisy)"

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
                                f"(|r|={corr.loc[col, other]:.3f} > {self.corr_threshold})"
                            )

        self._drop_set = drop_set
        self.keep_cols_ = [c for c in all_cols if c not in drop_set]
        self.n_kept_ = len(self.keep_cols_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._drop_set is None:
            raise RuntimeError("Call fit() before transform().")
        cols = [c for c in self.keep_cols_ if c in X.columns]
        return X[cols]

    def fit_transform(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
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


def preprocess_df_for_model(model_name: str,
                             data_name: str,
                             X_train: pd.DataFrame,
                             X_test: pd.DataFrame,
                             treatment_plans: list = None,
                             verbose: bool = False):
    """
    Column-selection preprocessing for the fold loop (or fixed OOD split).
    Called once per model per fold. Tree-based models receive unchanged
    DataFrames.
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


def balance_training_arms(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    treatment_plans: list,
    max_ratio: float = 3.0,
    seed: int = 42,
) -> tuple:
    """
    Downsample majority treatment arms so no arm exceeds
    `max(5, median_arm_size * max_ratio)` training samples.

    Only downsamples -- never synthesises data -- so no leakage is
    introduced. The median is computed from the current training fold
    (fit-on-train safe).
    """
    rng = np.random.RandomState(seed)

    arm_sizes = {tp: int(X_train[tp].sum()) for tp in treatment_plans
                 if tp in X_train.columns}
    nonempty = [s for s in arm_sizes.values() if s > 0]

    if not nonempty:
        return X_train, y_train

    median_sz = float(np.median(nonempty))
    cap = max(5, int(median_sz * max_ratio))

    keep_idx = []
    for tp, size in arm_sizes.items():
        idxs = X_train.index[X_train[tp] == 1].tolist()
        if size > cap:
            idxs = rng.choice(idxs, cap, replace=False).tolist()
            print(f"      [Balance] {tp}: {size} → {cap} samples "
                  f"(cap = {max_ratio}x median {median_sz:.0f})")
        keep_idx.extend(idxs)

    keep_idx_sorted = sorted(keep_idx)
    return X_train.loc[keep_idx_sorted], y_train.loc[keep_idx_sorted]
