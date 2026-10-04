# -*- coding: utf-8 -*-
"""
config.py
=========
Single source of truth for:
  - Environment / runtime setup (HF/TabPFN offline flags, CUDA checks, SEED)
  - Dataset registry: every dataset's files, protocol (CV / OOD / splits),
    outcome and its direction, recovery column, treatment plans, columns to
    remove, and naming/plotting metadata

This module has NO knowledge of any specific method (FDR or baselines) and
NO knowledge of evaluation/plotting. It only prepares data and describes the
experimental scenarios. fdr.py / baselines.py / evaluation.py / 
run_experiments.py all import from here.
"""

import os
import warnings
import numpy as np
import pandas as pd

# torch is only needed to pick FDR's TabPFN device. It is imported lazily and
# optionally so that torch-free baselines (e.g. CTR) can run even when torch
# is missing or broken in the environment.
try:
    import torch
    TORCH_AVAILABLE = True
    _TORCH_IMPORT_ERROR = None
except Exception as _torch_err:  # ImportError, or a broken/partial install
    torch = None
    TORCH_AVAILABLE = False
    _TORCH_IMPORT_ERROR = _torch_err
TORCH_METHODS = {"FDR", "BITES"}

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


# Default: <repo>/tabpfn/ (see README); override with the TABPFN_CKPT_PATH env var.
TABPFN_CKPT_PATH = os.environ.get(
    "TABPFN_CKPT_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "tabpfn",
                 "tabpfn-v3-regressor-v3_default.ckpt"),
)

# Seeds for the repeated runs: every CV dataset is cross-validated once per
# seed, and every OOD dataset is re-trained and re-evaluated once per seed.
SEED = 42
REPEAT_SEEDS = [42, 43, 44, 45, 46]

# FDR's OOD variant averages an ensemble of TabPFN fits in every repeat.
# Members get seeds derived from the repeat's seed, so repeats differ.
FDR_OOD_ENSEMBLE_SIZE = 5

def set_seed(seed: int):

    """Update the module-level SEED read by every model factory at call time."""
    global SEED
    SEED = seed


# ============================================================
# Startup checks: TabPFN checkpoint + CUDA
# ============================================================

def check_ckpt():
    if not os.path.isfile(TABPFN_CKPT_PATH):
        # Only FDR needs the checkpoint; warn here and let FDR fail if it runs.
        print(f"  [WARN] TabPFN checkpoint not found: '{TABPFN_CKPT_PATH}' -- FDR will fail; "
              "baselines are unaffected.")
        return
    if not os.access(TABPFN_CKPT_PATH, os.R_OK):
        raise PermissionError(f"TabPFN checkpoint is not readable: '{TABPFN_CKPT_PATH}'")
    size_mb = os.path.getsize(TABPFN_CKPT_PATH) / 1024 / 1024
    print(f"  [INFO] TabPFN checkpoint OK: '{TABPFN_CKPT_PATH}' ({size_mb:.0f} MB)")


def check_cuda():
    if torch is None:
        print(f"  [WARN] torch could not be imported ({_TORCH_IMPORT_ERROR!r}); "
              "FDR and BITES will be skipped; all other methods still run.")
        return "cpu"
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
# Dataset registry -- EVERY dataset is declared here
# ============================================================
#
# To add a dataset: put its CSV(s) in input/<name>/ and add one entry below
# (copy the closest existing one). Nothing else needs to change: the
# lookup tables further down, the runner and the evaluation all read these
# entries. `python run_experiments.py --list-datasets` shows what is
# registered.
#
# Evaluation protocols ("scenario"):
#   "cv"     : repeated stratified k-fold CV on files["data"], once per seed
#              in REPEAT_SEEDS (in-distribution).
#   "ood"    : fixed train/test files (test cohort != train cohort), run once
#              per seed in REPEAT_SEEDS; the split stays fixed and each repeat
#              re-trains every method with a different random seed.
#   "splits" : several predefined train/test pairs, files["train"] /
#              files["test"] containing "{k}", k = 1..files["n_splits"].
#
# Keys of an entry:
#   enabled           False = keep the definition but do not run/evaluate it
#   scenario          "cv" | "ood" | "splits"
#   files             file names inside input/<input_dir or name>/
#   outcome_col       outcome the methods model (RCB.score, pCR, ...)
#   lower_outcome_is_better   True for RCB-like, False for pCR-like outcomes
#   recovery_col      binary success used by CAU / RRD / Recovery Ratio
#   treatment_plans   one-hot treatment columns (2 or more)
#   remove_cols       columns never used as features (ids, other outcomes,
#                     anything measured after treatment)
#   id_col            patient id (patient-level bootstrap / permutation)
#   display_name, title    table row label, figure title
#   group             datasets sharing outcome + treatments; cross-dataset
#                     means and multiple-testing correction are per group
#   cohort, modality  a cohort with one "Clinical" and one "Multi-omics"
#                     dataset forms a clinical-vs-multi-omics pair
#   input_dir         input sub-folder if different from the name (optional)
#   cv_k              CV folds (optional, default 5)
# Column-selection settings for FDR stay in PREPROCESS_CONFIGS (below).

SCENARIOS = ("cv", "ood", "splits")

# Shared by all TransNEO / ARTemis datasets.
_NEOADJUVANT_TP = ["TP1", "TP2", "TP3", "TP4"]
_NEOADJUVANT_COMMON = {
    "outcome_col": "RCB.score",
    "lower_outcome_is_better": True,
    "recovery_col": "resp.pCR",
    "treatment_plans": _NEOADJUVANT_TP,
    "id_col": "Trial.ID",
    "group": "TransNEO_ARTemis",
}
_NEOADJUVANT_REMOVE = [
    "Trial.ID", "resp.Chemosensitive", "resp.Chemoresistant",
    "resp.pCR", "RCB.category",
    "Chemo.NumCycles", "Chemo.first.Taxane", "Chemo.first.Anthracycline",
    "Chemo.second.Taxane", "Chemo.second.Anthracycline",
    "Chemo.any.Anthracycline", "Chemo.any.antiHER2",
]

DATASET_REGISTRY = {
    "clin_TransNEO": {
        **_NEOADJUVANT_COMMON,
        "enabled": True,
        "scenario": "cv",
        "files": {"data": "clin_TransNEO.csv"},
        "remove_cols": list(_NEOADJUVANT_REMOVE),
        "display_name": "TransNEO clinical",
        "title": "TransNEO Clinical Dataset",
        "cohort": "TransNEO", "modality": "Clinical",
    },
    "clin_ARTemis": {
        **_NEOADJUVANT_COMMON,
        "enabled": True,
        "scenario": "cv",
        "files": {"data": "clin_ARTemis.csv"},
        "remove_cols": list(_NEOADJUVANT_REMOVE),
        "display_name": "ARTemis clinical",
        "title": "ARTemis Clinical Dataset",
        "cohort": "ARTemis", "modality": "Clinical",
    },
    "multi_Trans_ART": {
        **_NEOADJUVANT_COMMON,
        "enabled": True,
        "scenario": "cv",
        "files": {"data": "multi_Trans_ART.csv"},
        "remove_cols": list(_NEOADJUVANT_REMOVE),
        "display_name": "Combined TransNEO + ARTemis multi-omics CV",
        "title": "Multi-omics Datasets CV: TransNEO and ARTemis",
        "cohort": "TransNEO+ARTemis", "modality": "Multi-omics",
    },
    "multi_TransNEO": {
        **_NEOADJUVANT_COMMON,
        "enabled": True,
        "scenario": "cv",
        "files": {"data": "multi_TransNEO.csv"},
        "remove_cols": list(_NEOADJUVANT_REMOVE),
        "display_name": "TransNEO multi-omics",
        "title": "TransNEO Multi-omics Dataset",
        "cohort": "TransNEO", "modality": "Multi-omics",
    },
    "multi_ARTemis": {
        **_NEOADJUVANT_COMMON,
        "enabled": True,
        "scenario": "cv",
        "files": {"data": "multi_ARTemis.csv"},
        "remove_cols": list(_NEOADJUVANT_REMOVE),
        "display_name": "ARTemis multi-omics",
        "title": "ARTemis Multi-omics Dataset",
        "cohort": "ARTemis", "modality": "Multi-omics",
    },
    # ── OOD scenario: fixed train/test split, test cohort != train cohort,
    #    repeated once per seed in REPEAT_SEEDS ──
    "OOD_multi_Trans_ART": {
        **_NEOADJUVANT_COMMON,
        "enabled": True,
        "scenario": "ood",
        "files": {"train": "OOD_multi_Trans_ART_train.csv",
                  "test": "OOD_multi_Trans_ART_test.csv"},
        "remove_cols": list(_NEOADJUVANT_REMOVE),
        "display_name": "TransNEO and ARTemis multi-omics OOD",
        "title": "Multi-omics Datasets OOD: TransNEO train, ARTemis test",
        "cohort": "TransNEO->ARTemis", "modality": "Multi-omics",
    },

}


def _validate_registry(registry: dict) -> dict:
    """Check every entry and fill optional keys; returns the ENABLED entries."""
    required = ["scenario", "files", "outcome_col", "lower_outcome_is_better",
                "recovery_col", "treatment_plans", "remove_cols"]
    need_files = {"cv": ["data"], "ood": ["train", "test"], "splits": ["train", "test", "n_splits"]}
    enabled = {}
    for name, spec in registry.items():
        missing = [k for k in required if k not in spec]
        if missing:
            raise ValueError(f"DATASET_REGISTRY['{name}'] is missing keys: {missing}")
        if spec["scenario"] not in SCENARIOS:
            raise ValueError(f"DATASET_REGISTRY['{name}']: scenario must be one of {SCENARIOS}")
        miss_f = [k for k in need_files[spec["scenario"]] if k not in spec["files"]]
        if miss_f:
            raise ValueError(f"DATASET_REGISTRY['{name}']: scenario '{spec['scenario']}' "
                             f"needs files {miss_f}")
        if len(spec["treatment_plans"]) < 2:
            raise ValueError(f"DATASET_REGISTRY['{name}']: need at least 2 treatment_plans")
        spec.setdefault("enabled", True)
        spec.setdefault("display_name", name)
        spec.setdefault("title", spec["display_name"])
        spec.setdefault("group", name)
        spec.setdefault("cohort", None)
        spec.setdefault("modality", None)
        spec.setdefault("id_col", None)
        spec.setdefault("input_dir", name)
        spec.setdefault("cv_k", 5)
        spec.setdefault("outcome_label", spec["outcome_col"])
        if spec["enabled"]:
            enabled[name] = spec
    if not enabled:
        raise ValueError("No enabled datasets in DATASET_REGISTRY.")
    return enabled


# Everything below uses only the ENABLED datasets.
ALL_DATASET_DEFINITIONS = DATASET_REGISTRY
DATASET_REGISTRY = _validate_registry(DATASET_REGISTRY)


def dataset_input_dir(data_name: str, base_path: str = None) -> str:
    base_path = base_path or os.getcwd()
    return os.path.join(base_path, "input", DATASET_REGISTRY[data_name]["input_dir"])


def outcome_sign(data_name: str) -> float:
    """+1 when a lower outcome is better (RCB), -1 when higher is better (pCR).
    Every method minimises `outcome_sign * outcome`."""
    return 1.0 if DATASET_REGISTRY[data_name]["lower_outcome_is_better"] else -1.0


def recovery_col(data_name: str) -> str:
    return DATASET_REGISTRY[data_name]["recovery_col"]


def id_col(data_name: str):
    return DATASET_REGISTRY[data_name].get("id_col")


# ---- Lookup tables derived from the registry ----

CV_DATASETS     = [d for d, c in DATASET_REGISTRY.items() if c["scenario"] == "cv"]
OOD_DATASETS    = [d for d, c in DATASET_REGISTRY.items() if c["scenario"] == "ood"]
SPLITS_DATASETS = [d for d, c in DATASET_REGISTRY.items() if c["scenario"] == "splits"]

DATASET_NAME_MAP = {d: c["display_name"] for d, c in DATASET_REGISTRY.items()}
DATASET_TITLES   = {d: c["title"] for d, c in DATASET_REGISTRY.items()}
DATASET_GROUPS   = {d: c["group"] for d, c in DATASET_REGISTRY.items()}

# Row order of tables and figures (independent of the processing order above).
DATASET_ROW_ORDER_KEYS = [
    "clin_TransNEO", "clin_ARTemis", "multi_TransNEO", "multi_ARTemis",
    "multi_Trans_ART", "OOD_multi_Trans_ART",
]
DATASET_ROW_ORDER = (
    [DATASET_NAME_MAP[d] for d in DATASET_ROW_ORDER_KEYS if d in DATASET_NAME_MAP]
    + [DATASET_NAME_MAP[d] for d in DATASET_NAME_MAP if d not in DATASET_ROW_ORDER_KEYS]
)

# Within-cohort Clinical vs Multi-omics pairs: a cohort with one "Clinical"
# and one "Multi-omics" dataset forms a pair automatically.
CLINICAL_MULTIOMICS_PAIRS = {}
for _d, _c in DATASET_REGISTRY.items():
    if _c["cohort"] and _c["modality"] in ("Clinical", "Multi-omics"):
        _key = "clinical" if _c["modality"] == "Clinical" else "multiomics"
        CLINICAL_MULTIOMICS_PAIRS.setdefault(_c["cohort"], {})[_key] = _d
CLINICAL_MULTIOMICS_PAIRS = {k: v for k, v in CLINICAL_MULTIOMICS_PAIRS.items()
                             if set(v) == {"clinical", "multiomics"}}

# Modality is only used for the paired Clinical-vs-Multi-omics comparison.
DATASET_MODALITY = {d: DATASET_REGISTRY[d]["modality"]
                    for pair in CLINICAL_MULTIOMICS_PAIRS.values() for d in pair.values()}

MULTIOMICS_DATASETS = [d for d, c in DATASET_REGISTRY.items() if c["modality"] == "Multi-omics"]


# ============================================================
# Method registry metadata (names/colours/grouping for plots & tables)
# ============================================================
#
# 11 methods (FDR + 10 baselines)

METHOD_META = {
    "FDR":   ("FDR",           "Proposed"),
    "CB":    ("CatBoost",      "A. Classical"),
    "XGB":   ("XGBoost",       "A. Classical"),
    "S_L":   ("S-Learner",     "B. Meta-Learner"),
    "X_L":   ("X-Learner",     "B. Meta-Learner"),
    "DR_L":  ("DR-Learner",    "B. Meta-Learner"),
    "R_L":   ("R-Learner",     "B. Meta-Learner"),
    "CF":    ("Causal Forest", "C. Causal"),
    "CTR":   ("CTR (Causal Tree)", "C. Causal"),
    "CUTS":  ("CUTS",          "D. Modern SOTA"),
    "BITES": ("BITES",         "D. Modern SOTA"),
}

ALL_METHOD_COL_ORDER = [
    "FDR", "CB", "XGB",
    "S_L", "X_L", "DR_L", "R_L",
    "CF", "CTR", "CUTS", "BITES",
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
    "CTR":   "olive",
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
