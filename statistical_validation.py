# -*- coding: utf-8 -*-
"""
statistical_validation.py
==========================
Statistical validation of the recommendation metrics produced by
evaluation.py. Every function here is called from evaluation.py.

Contents
--------
1. Patient-level bootstrap CIs for CAU / RRD / Recovery Ratio
   (bootstrap_ci_recovery_metrics) and a cross-dataset bootstrap CI of the
   per-method mean (bootstrap_ci_across_datasets).
2. Permutation tests of CAU (permutation_test_recovery_metrics) and of the
   doubly-robust policy uplift (permutation_test_dr_uplift).
3. Off-policy evaluation of each method's policy on RCB.score:
   IPW (self-normalised) and doubly-robust (DR) policy value, their uplift
   over the observed standard of care, and a bootstrap CI for DR uplift.
4. Multiplicity correction (Bonferroni + Benjamini-Hochberg).
5. Covariate balance (standardised mean differences, SMD).
6. IPW-adjusted CAU (sensitivity check against the naive CAU).
7. E-values for unmeasured confounding (VanderWeele & Ding, 2017).

Important: repeated-CV REC files pool the SAME patients over several runs
------------------------------------------------------------------------
For CV datasets a method's REC file stacks N_Runs repeated 5-fold CV runs,
so every patient appears N_Runs times. Resampling or permuting ROWS would
treat those copies as independent patients and make CIs too narrow and
p-values too small. All resampling below is therefore done at the PATIENT
level (clusters identified by `Trial.ID`; falling back to the row position
within a run). A bootstrap draw resamples patients and keeps all of that
patient's rows; a permutation shuffles which arm each PATIENT received and
applies the same shuffle in every run.

Where the results go (written by evaluation.py)
-----------------------------------------------
  output/<dataset>/<dataset>_Statistical_Validation.csv
        one row per method: patient-level bootstrap CIs, permutation
        p-values, IPW-adjusted CAU, E-values, IPW/DR policy value & uplift.
  output/output_StatisticalValidation_AllDatasets.csv
        the same for every dataset, plus Bonferroni / BH-adjusted p-values.

  The CIs and the CAU permutation p-values are also merged into
  output/output_AllDatasets_Metrics.csv and the per-dataset
  <dataset>_Recovery_Metrics_summary.csv.
"""

import zlib

import numpy as np
import pandas as pd

import config

# ============================================================
# Global settings
# ============================================================

N_BOOTSTRAP = 1000
N_PERMUTATIONS = 1000
BOOTSTRAP_SEED = 12345
CI_LEVEL = 0.95
PROPENSITY_CLIP = 1e-3

# Methods whose per-arm REC columns hold predicted TREATMENT EFFECTS rather
# than predicted OUTCOMES. The DR estimator needs predicted outcomes, so for
# these methods only the IPW policy value is reported (DR is NaN).
NON_OUTCOME_PREDICTION_METHODS = {"CTR"}

RECOVERY_METRICS = ["Recovery_Ratio", "RRD", "RRD_pp", "CAU", "CAU_pp"]


# ============================================================
# 0. Helpers: patient clusters, prediction columns, arm indices
# ============================================================

def _patient_codes(df: pd.DataFrame) -> np.ndarray:
    """Integer patient id per row (0..K-1). Uses Trial.ID when present,
    otherwise the row's position within its run (each CV run lists every
    patient exactly once), otherwise the row index."""
    if "Trial.ID" in df.columns and df["Trial.ID"].notna().all():
        ids = df["Trial.ID"].astype(str).values
    elif "Run" in df.columns:
        ids = df.groupby("Run").cumcount().values
    else:
        ids = np.arange(len(df))
    codes, _ = pd.factorize(ids)
    return codes


def _stable_seed(seed: int, *keys) -> int:
    """Deterministic per-key seed, so results for one method/dataset do not
    depend on which other methods/datasets were evaluated (or their order)."""
    h = zlib.crc32("|".join(str(k) for k in keys).encode("utf-8"))
    return int((seed + h) % (2**31 - 1))


def _cluster_bootstrap_row_weights(codes: np.ndarray, n_clusters: int, rng) -> np.ndarray:
    """Resample patients with replacement; return how many times each ROW is
    included (= how many times its patient was drawn)."""
    draws = rng.randint(0, n_clusters, size=n_clusters)
    cluster_counts = np.bincount(draws, minlength=n_clusters)
    return cluster_counts[codes].astype(float)


def _patient_level_permutation(codes, patient_value, rng):
    """Shuffle a per-patient attribute across patients; return per-row values."""
    perm = rng.permutation(len(patient_value))
    return patient_value[perm][codes]


def _per_patient_constant(codes, values):
    """Return (per-patient array, ok) where ok=False if a patient has more
    than one distinct value (then patient-level permutation is not valid)."""
    s = pd.Series(values).groupby(codes)
    ok = bool((s.nunique(dropna=False) <= 1).all())
    return s.first().sort_index().values, ok


def prediction_columns(df: pd.DataFrame, treatment_plans: list):
    """
    Columns holding each method's per-arm predictions in a REC file.

    REC files contain the treatment-plan names twice: first the observed
    one-hot indicators (from the original data), then the method's per-arm
    predictions. pandas renames the second copy to '<tp>.1' when reading the
    CSV, so those are the prediction columns. Returns None if not found.
    """
    cols = [f"{tp}.1" for tp in treatment_plans]
    if all(c in df.columns for c in cols):
        return cols
    return None


def _arm_indices(df, treatment_plans, col):
    mapping = {tp: i for i, tp in enumerate(treatment_plans)}
    return df[col].map(mapping).values.astype(float)  # NaN if unknown


# ============================================================
# 1. Bootstrap confidence intervals
# ============================================================

def _recovery_metrics(follow, pcr, w):
    """CAU/RRD/Recovery Ratio from (possibly bootstrap-weighted) rows."""
    n_follow = np.sum(w * follow)
    n_not = np.sum(w * ~follow)
    if n_follow <= 0 or n_not <= 0:
        return {k: np.nan for k in RECOVERY_METRICS}
    p_follow = np.sum(w * follow * pcr) / n_follow
    p_not = np.sum(w * ~follow * pcr) / n_not
    coverage = n_follow / (n_follow + n_not)
    rrd = p_follow - p_not
    cau = rrd * coverage
    return {
        "Recovery_Ratio": (p_follow / p_not) if p_not > 0 else np.nan,
        "RRD": rrd, "RRD_pp": rrd * 100.0,
        "CAU": cau, "CAU_pp": cau * 100.0,
    }


def bootstrap_ci_recovery_metrics(
    df: pd.DataFrame,
    outcome_col: str,
    n_boot: int = N_BOOTSTRAP,
    ci_level: float = CI_LEVEL,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """
    Patient-level (cluster) bootstrap percentile CIs for Recovery_Ratio,
    CAU, CAU_pp, RRD and RRD_pp.

    Each draw resamples PATIENTS with replacement and keeps all of each
    drawn patient's rows (one per repeated-CV run), then recomputes the
    metric on the pooled rows. This captures sampling variability of the
    actual cohort (N patients), not of the N x N_Runs pooled rows.

    Also returns `<metric>_observed` (the point estimate on the full data)
    and `<metric>_boot_mean`.
    """
    n = len(df)
    if n == 0 or "resp.pCR" not in df.columns:
        return {"ci_level": ci_level, "n_boot": n_boot, "n": n}

    rng = np.random.RandomState(seed)
    follow = df["FOLLOW_REC"].astype(bool).values
    pcr = df["resp.pCR"].astype(float).values
    codes = _patient_codes(df)
    k = codes.max() + 1

    observed = _recovery_metrics(follow, pcr, np.ones(n))
    boot = {m: np.empty(n_boot) for m in RECOVERY_METRICS}
    for b in range(n_boot):
        w = _cluster_bootstrap_row_weights(codes, k, rng)
        vals = _recovery_metrics(follow, pcr, w)
        for m in RECOVERY_METRICS:
            boot[m][b] = vals[m]

    alpha = 1 - ci_level
    result = {"ci_level": ci_level, "n_boot": n_boot, "n": n, "n_patients": int(k)}
    for m, arr in boot.items():
        finite = arr[np.isfinite(arr)]
        result[f"{m}_observed"] = float(observed[m]) if np.isfinite(observed[m]) else np.nan
        if len(finite) == 0:
            result[f"{m}_ci_lo"] = result[f"{m}_ci_hi"] = result[f"{m}_boot_mean"] = np.nan
        else:
            result[f"{m}_ci_lo"] = float(np.percentile(finite, 100 * alpha / 2))
            result[f"{m}_ci_hi"] = float(np.percentile(finite, 100 * (1 - alpha / 2)))
            result[f"{m}_boot_mean"] = float(np.mean(finite))
        result[f"{m}_n_valid_boot"] = int(len(finite))
    return result


def bootstrap_ci_across_datasets(
    per_dataset_values: pd.DataFrame,
    value_col: str,
    method_col: str = "Method",
    dataset_col: str = "DataSet",
    n_boot: int = N_BOOTSTRAP,
    ci_level: float = CI_LEVEL,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """
    Cross-dataset bootstrap CI: for each method, resamples DATASETS with
    replacement and recomputes the across-dataset mean of `value_col`,
    giving a 95% percentile CI on "the mean of this metric across all
    datasets". `per_dataset_values` has one row per (dataset, method).

    Each method uses its own deterministic random stream, so its CI does not
    change when other methods are added to or removed from the comparison.
    """
    rows = []
    alpha = 1 - ci_level
    for method, g in per_dataset_values.groupby(method_col):
        vals = g[value_col].dropna().values
        n = len(vals)
        if n == 0:
            continue
        rng = np.random.RandomState(_stable_seed(seed, "across_datasets", value_col, method))
        boot_means = np.array([np.mean(vals[rng.randint(0, n, size=n)]) for _ in range(n_boot)])
        rows.append({
            "Method": method,
            "N_Datasets": n,
            f"{value_col}_mean": float(np.mean(vals)),
            f"{value_col}_std": float(np.std(vals, ddof=1)) if n > 1 else 0.0,
            f"{value_col}_ci95_lo": float(np.percentile(boot_means, 100 * alpha / 2)),
            f"{value_col}_ci95_hi": float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(f"{value_col}_mean", ascending=False).reset_index(drop=True)
    return out


# ============================================================
# 2. Permutation tests
# ============================================================

def _permutation_p_values(observed, null_vals):
    finite = null_vals[np.isfinite(null_vals)]
    if len(finite) == 0 or not np.isfinite(observed):
        return np.nan, np.nan, np.nan, np.nan, 0
    null_mean = float(np.mean(finite))
    # Two-sided, centred on the null mean (the null of CAU/RRD is not exactly 0).
    p_two = (np.sum(np.abs(finite - null_mean) >= abs(observed - null_mean)) + 1) / (len(finite) + 1)
    # One-sided in the direction of benefit (larger CAU/RRD/RR = better).
    p_greater = (np.sum(finite >= observed) + 1) / (len(finite) + 1)
    return float(p_two), float(p_greater), null_mean, float(np.std(finite)), int(len(finite))


def permutation_test_recovery_metrics(
    df: pd.DataFrame,
    outcome_col: str,
    n_perm: int = N_PERMUTATIONS,
    seed: int = BOOTSTRAP_SEED,
    metric: str = "CAU_pp",
) -> dict:
    """
    Permutation test of the null "the recommendation carries no information
    about who recovers". The arm each PATIENT actually received (CURRENT_TP)
    is shuffled across patients -- the same shuffle in every repeated-CV run
    -- independently of REC_TP and outcome. This preserves the arm
    proportions, the outcome distribution and the within-patient
    correlation across runs.

    Returns:
      p_value          two-sided p (distance from the null mean)
      p_value_greater  one-sided p for "metric is larger than by chance"
                       (the natural test for a recommendation that helps)
    """
    n = len(df)
    if (n == 0 or "CURRENT_TP" not in df.columns or "REC_TP" not in df.columns
            or "resp.pCR" not in df.columns):
        return {"metric": metric, "p_value": np.nan, "p_value_greater": np.nan, "n_perm": n_perm, "n": n}

    rng = np.random.RandomState(seed)
    rec_tp = df["REC_TP"].astype(str).values
    current_tp = df["CURRENT_TP"].astype(str).values
    pcr = df["resp.pCR"].astype(float).values
    ones = np.ones(n)
    codes = _patient_codes(df)
    patient_tp, patient_level = _per_patient_constant(codes, current_tp)

    observed = _recovery_metrics(rec_tp == current_tp, pcr, ones)[metric]
    null_vals = np.empty(n_perm)
    for i in range(n_perm):
        perm_tp = (_patient_level_permutation(codes, patient_tp, rng)
                   if patient_level else rng.permutation(current_tp))
        null_vals[i] = _recovery_metrics(rec_tp == perm_tp, pcr, ones)[metric]

    p_two, p_greater, null_mean, null_std, n_valid = _permutation_p_values(observed, null_vals)
    return {
        "metric": metric,
        "observed": float(observed) if np.isfinite(observed) else np.nan,
        "null_mean": null_mean, "null_std": null_std,
        "p_value": p_two, "p_value_greater": p_greater,
        "n_perm": n_perm, "n_valid_perm": n_valid, "n": n,
        "patient_level_permutation": patient_level,
    }


# ============================================================
# 3. Off-policy evaluation: IPW and doubly-robust policy value
# ============================================================
#
# Outcome is RCB.score (LOWER = better). "Standard of care" (SoC) is the
# observed treatment policy, whose value is simply the mean observed
# outcome. Uplift = policy value - SoC, so NEGATIVE uplift = the policy is
# estimated to reduce residual cancer burden.

def compute_ipw_value(y, actual_idx, propensity, policy_idx, weights=None, clip=PROPENSITY_CLIP):
    """Self-normalised (Hajek) IPW value of the policy."""
    w_row = np.ones(len(y)) if weights is None else weights
    valid = np.isfinite(actual_idx) & np.isfinite(policy_idx) & np.isfinite(y)
    a = actual_idx[valid].astype(int)
    match = (a == policy_idx[valid].astype(int)).astype(float)
    e = np.clip(propensity[valid, a], clip, 1.0)
    w = w_row[valid] * match / e
    if w.sum() <= 0:
        return {"value": np.nan, "ess": np.nan, "n_matched": 0}
    value = np.sum(w * y[valid]) / np.sum(w)
    ess = w.sum() ** 2 / np.sum(w ** 2)
    return {"value": float(value), "ess": float(ess), "n_matched": int(np.sum(match * (w_row[valid] > 0)))}


def compute_dr_value(y, actual_idx, propensity, policy_idx, q_hat, weights=None, clip=PROPENSITY_CLIP):
    """Doubly-robust (AIPW) value: mean[ q(x, pi(x)) + 1{A=pi(x)}/e_A(x) * (y - q(x, A)) ]."""
    w_row = np.ones(len(y)) if weights is None else weights
    valid = (np.isfinite(actual_idx) & np.isfinite(policy_idx) & np.isfinite(y)
             & np.all(np.isfinite(q_hat), axis=1))
    if not np.any(valid):
        return {"value": np.nan, "ess": np.nan, "n_matched": 0}
    a = actual_idx[valid].astype(int)
    pi = policy_idx[valid].astype(int)
    rows = np.arange(valid.sum())
    q = q_hat[valid]
    match = (a == pi).astype(float)
    e = np.clip(propensity[valid, a], clip, 1.0)
    ipw = match / e
    psi = q[rows, pi] + ipw * (y[valid] - q[rows, a])
    wr = w_row[valid]
    value = np.sum(wr * psi) / np.sum(wr)
    wi = wr * ipw
    ess = (wi.sum() ** 2 / np.sum(wi ** 2)) if np.sum(wi ** 2) > 0 else np.nan
    return {"value": float(value), "ess": float(ess), "n_matched": int(np.sum(match * (wr > 0)))}


def compute_standard_of_care_value(y, weights=None):
    """Value of the observed (standard-of-care) policy = mean observed outcome."""
    w = np.ones(len(y)) if weights is None else weights
    ok = np.isfinite(y)
    return float(np.sum(w[ok] * y[ok]) / np.sum(w[ok]))


def _policy_arrays(df, treatment_plans, outcome_col, propensity=None, q_cols="auto",
                   rec_col="REC_TP", current_col="CURRENT_TP"):
    y = pd.to_numeric(df[outcome_col], errors="coerce").values.astype(float)
    actual_idx = _arm_indices(df, treatment_plans, current_col)
    policy_idx = _arm_indices(df, treatment_plans, rec_col)
    if propensity is None:
        propensity = df[[f"PROP_{tp}" for tp in treatment_plans]].values.astype(float)
    if q_cols == "auto":
        q_cols = prediction_columns(df, treatment_plans)
    q_hat = df[q_cols].astype(float).values if q_cols else None
    return y, actual_idx, policy_idx, propensity, q_hat


def _policy_value_from_arrays(y, actual_idx, policy_idx, propensity, q_hat, weights=None):
    ipw = compute_ipw_value(y, actual_idx, propensity, policy_idx, weights)
    soc = compute_standard_of_care_value(y, weights)
    if q_hat is not None:
        dr = compute_dr_value(y, actual_idx, propensity, policy_idx, q_hat, weights)
    else:
        dr = {"value": np.nan, "ess": np.nan, "n_matched": np.nan}
    return {
        "IPW_value": ipw["value"], "IPW_ess": ipw["ess"], "IPW_n_matched": ipw["n_matched"],
        "DR_value": dr["value"], "DR_ess": dr["ess"], "DR_n_matched": dr["n_matched"],
        "SoC_value": soc,
        "IPW_uplift": ipw["value"] - soc,   # negative = lower RCB = improvement
        "DR_uplift": dr["value"] - soc,     # negative = lower RCB = improvement
    }


def evaluate_policy_value(
    df: pd.DataFrame,
    treatment_plans: list,
    outcome_col: str,
    propensity: np.ndarray = None,
    q_cols="auto",
    rec_col: str = "REC_TP",
    current_col: str = "CURRENT_TP",
) -> dict:
    """
    IPW and DR value of a method's recommendation policy on `outcome_col`
    (RCB.score), plus their uplift over the observed standard of care.

    `propensity` : (n, n_arms) P(actual arm | x); default = PROP_<tp> columns.
    `q_cols`     : per-arm predicted-OUTCOME columns for DR. "auto" uses the
                   '<tp>.1' prediction columns of a REC file; None -> IPW only.
    """
    arrays = _policy_arrays(df, treatment_plans, outcome_col, propensity, q_cols, rec_col, current_col)
    out = _policy_value_from_arrays(*arrays)
    out["N"] = len(df)
    return out


def bootstrap_ci_dr_uplift(
    df: pd.DataFrame,
    treatment_plans: list,
    outcome_col: str,
    n_boot: int = N_BOOTSTRAP,
    ci_level: float = CI_LEVEL,
    seed: int = BOOTSTRAP_SEED,
    q_cols="auto",
) -> dict:
    """Patient-level bootstrap CIs for IPW_uplift and DR_uplift."""
    n = len(df)
    empty = {"ci_level": ci_level, "n_boot": n_boot, "n": n,
             "DR_uplift_ci_lo": np.nan, "DR_uplift_ci_hi": np.nan,
             "IPW_uplift_ci_lo": np.nan, "IPW_uplift_ci_hi": np.nan}
    if n == 0 or not all(f"PROP_{tp}" in df.columns for tp in treatment_plans):
        return empty

    rng = np.random.RandomState(seed)
    arrays = _policy_arrays(df, treatment_plans, outcome_col, None, q_cols)
    codes = _patient_codes(df)
    k = codes.max() + 1

    boot = {"DR_uplift": np.empty(n_boot), "IPW_uplift": np.empty(n_boot)}
    for b in range(n_boot):
        w = _cluster_bootstrap_row_weights(codes, k, rng)
        pv = _policy_value_from_arrays(*arrays, weights=w)
        boot["DR_uplift"][b] = pv["DR_uplift"]
        boot["IPW_uplift"][b] = pv["IPW_uplift"]

    alpha = 1 - ci_level
    result = {"ci_level": ci_level, "n_boot": n_boot, "n": n}
    for m, arr in boot.items():
        finite = arr[np.isfinite(arr)]
        if len(finite) == 0:
            result.update({f"{m}_ci_lo": np.nan, f"{m}_ci_hi": np.nan, f"{m}_boot_mean": np.nan})
        else:
            result[f"{m}_ci_lo"] = float(np.percentile(finite, 100 * alpha / 2))
            result[f"{m}_ci_hi"] = float(np.percentile(finite, 100 * (1 - alpha / 2)))
            result[f"{m}_boot_mean"] = float(np.mean(finite))
        result[f"{m}_n_valid_boot"] = int(len(finite))
    return result


def permutation_test_dr_uplift(
    df: pd.DataFrame,
    treatment_plans: list,
    outcome_col: str,
    n_perm: int = N_PERMUTATIONS,
    seed: int = BOOTSTRAP_SEED,
    q_cols="auto",
    metric: str = "DR_uplift",
) -> dict:
    """
    Permutation test for DR_uplift (or IPW_uplift) against the null "the
    policy's recommended arm carries no information about outcome": the arm
    each PATIENT received is shuffled across patients (same shuffle in every
    run), leaving the propensity scores and per-arm predictions untouched.

    Returns the observed uplift (negative = improvement), a two-sided p and
    a one-sided p_value_less for "uplift is lower (better) than by chance".
    """
    n = len(df)
    if n == 0 or not all(f"PROP_{tp}" in df.columns for tp in treatment_plans):
        return {"metric": metric, "p_value": np.nan, "p_value_less": np.nan, "n_perm": n_perm, "n": n}

    rng = np.random.RandomState(seed)
    y, actual_idx, policy_idx, propensity, q_hat = _policy_arrays(
        df, treatment_plans, outcome_col, None, q_cols)
    observed = _policy_value_from_arrays(y, actual_idx, policy_idx, propensity, q_hat)[metric]
    if not np.isfinite(observed):
        return {"metric": metric, "observed": np.nan, "p_value": np.nan, "p_value_less": np.nan,
                "n_perm": n_perm, "n": n}

    codes = _patient_codes(df)
    patient_arm, patient_level = _per_patient_constant(codes, actual_idx)
    null_vals = np.empty(n_perm)
    for i in range(n_perm):
        perm_arm = (_patient_level_permutation(codes, patient_arm, rng)
                    if patient_level else rng.permutation(actual_idx))
        null_vals[i] = _policy_value_from_arrays(y, perm_arm, policy_idx, propensity, q_hat)[metric]

    # Benefit = LOWER uplift, so flip signs to reuse the "greater" helper.
    p_two, p_less, null_mean, null_std, n_valid = _permutation_p_values(-observed, -null_vals)
    return {
        "metric": metric, "observed": float(observed),
        "null_mean": -null_mean if np.isfinite(null_mean) else np.nan, "null_std": null_std,
        "p_value": p_two, "p_value_less": p_less,
        "n_perm": n_perm, "n_valid_perm": n_valid, "n": n,
    }


# ============================================================
# 4. Multiplicity correction
# ============================================================

def apply_multiplicity_correction(p_values_df: pd.DataFrame, p_col: str = "p_value", alpha: float = 0.05,
                                  prefix: str = "") -> pd.DataFrame:
    """
    Adds Bonferroni and Benjamini-Hochberg (BH) FDR-adjusted p-values and
    significance flags for the p-values in `p_col` (one row per comparison).
    Output columns are named `<prefix>bonferroni_p`, `<prefix>bh_p`, etc.
    """
    out = p_values_df.copy()
    p = pd.to_numeric(out[p_col], errors="coerce")
    valid = p.notna()
    m = int(valid.sum())
    out[f"{prefix}bonferroni_p"] = np.nan
    out[f"{prefix}bh_p"] = np.nan
    if m > 0:
        out.loc[valid, f"{prefix}bonferroni_p"] = (p[valid] * m).clip(upper=1.0)
        pv = p[valid].values
        order = np.argsort(pv)
        ranked = pv[order] * m / np.arange(1, m + 1)
        bh_sorted = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
        bh = np.empty(m)
        bh[order] = bh_sorted
        out.loc[valid, f"{prefix}bh_p"] = bh
    out[f"{prefix}bonferroni_significant"] = out[f"{prefix}bonferroni_p"] < alpha
    out[f"{prefix}bh_significant"] = out[f"{prefix}bh_p"] < alpha
    return out


# ============================================================
# 5. Covariate balance / Standardized Mean Difference (SMD)
# ============================================================

def compute_smd_table(X: pd.DataFrame, treatment_plans: list, weights: np.ndarray = None) -> pd.DataFrame:
    """
    Standardized Mean Difference (SMD) for every covariate, comparing each
    treatment arm against the pooled other arms:
        SMD = (mean_arm - mean_rest) / sqrt((var_arm + var_rest) / 2)
    |SMD| > 0.1 = meaningful and > 0.25 = substantial imbalance (Austin,
    2009). Optional `weights` (e.g. IPW) give the weighted SMD.
    """
    covariate_cols = [c for c in X.columns if c not in treatment_plans]
    rows = []
    w = weights if weights is not None else np.ones(len(X))

    for tp in treatment_plans:
        if tp not in X.columns:
            continue
        in_arm = X[tp].values == 1
        out_arm = ~in_arm
        if in_arm.sum() == 0 or out_arm.sum() == 0:
            continue
        for col in covariate_cols:
            vals = X[col].values.astype(float)
            ok = np.isfinite(vals)
            v_in, v_out = vals[in_arm & ok], vals[out_arm & ok]
            w_in, w_out = w[in_arm & ok], w[out_arm & ok]
            if len(v_in) == 0 or len(v_out) == 0:
                continue
            mean_in = np.average(v_in, weights=w_in)
            mean_out = np.average(v_out, weights=w_out)
            var_in = np.average((v_in - mean_in) ** 2, weights=w_in)
            var_out = np.average((v_out - mean_out) ** 2, weights=w_out)
            pooled_std = np.sqrt((var_in + var_out) / 2.0)
            smd = (mean_in - mean_out) / pooled_std if pooled_std > 0 else np.nan
            rows.append({
                "Arm": tp, "Covariate": col,
                "Mean_Arm": mean_in, "Mean_Rest": mean_out,
                "SMD": smd, "Abs_SMD": abs(smd) if np.isfinite(smd) else np.nan,
                "Weighted": weights is not None,
            })
    return pd.DataFrame(rows)


def summarize_smd_table(smd_df: pd.DataFrame) -> pd.DataFrame:
    """Per-arm count/proportion of covariates with |SMD| > 0.1 and > 0.25, and max |SMD|."""
    cols = ["Arm", "N_Covariates", "N_SMD_gt_0.1", "Pct_SMD_gt_0.1",
            "N_SMD_gt_0.25", "Pct_SMD_gt_0.25", "Max_Abs_SMD"]
    if smd_df.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for arm, g in smd_df.groupby("Arm"):
        n_cov = int(g["Abs_SMD"].notna().sum())
        n_gt_01 = int((g["Abs_SMD"] > 0.1).sum())
        n_gt_025 = int((g["Abs_SMD"] > 0.25).sum())
        rows.append({
            "Arm": arm, "N_Covariates": n_cov,
            "N_SMD_gt_0.1": n_gt_01, "Pct_SMD_gt_0.1": n_gt_01 / n_cov if n_cov else np.nan,
            "N_SMD_gt_0.25": n_gt_025, "Pct_SMD_gt_0.25": n_gt_025 / n_cov if n_cov else np.nan,
            "Max_Abs_SMD": float(g["Abs_SMD"].max()) if n_cov else np.nan,
        })
    return pd.DataFrame(rows, columns=cols)


# ============================================================
# 6. IPW-adjusted CAU (sensitivity check against the naive CAU)
# ============================================================

def compute_adjusted_cau(
    df: pd.DataFrame,
    treatment_plans: list,
    outcome_col: str,
    clip: float = PROPENSITY_CLIP,
) -> dict:
    """
    IPW-adjusted CAU / RRD / Recovery Ratio: each patient contributes
    1 / e_{actual arm}(x) instead of 1 to their follow / not-follow group.
    A large gap between adjusted and naive CAU signals that measured
    confounding drives the naive number. Requires PROP_<tp> columns.
    """
    nan = {"CAU_adj": np.nan, "CAU_pp_adj": np.nan, "RRD_adj": np.nan,
           "RRD_pp_adj": np.nan, "Recovery_Ratio_adj": np.nan, "IPW_ess_all": np.nan}
    if (not all(f"PROP_{tp}" in df.columns for tp in treatment_plans)
            or "resp.pCR" not in df.columns):
        return nan

    follow = df["FOLLOW_REC"].astype(bool).values
    actual_idx = _arm_indices(df, treatment_plans, "CURRENT_TP")
    valid = np.isfinite(actual_idx)
    propensity = df[[f"PROP_{tp}" for tp in treatment_plans]].values.astype(float)
    e_actual = np.full(len(df), np.nan)
    e_actual[valid] = propensity[valid, actual_idx[valid].astype(int)]
    ipw = 1.0 / np.clip(e_actual, clip, 1.0)
    ipw[~valid] = 0.0
    pcr = df["resp.pCR"].astype(float).values

    n_follow_w = ipw[follow].sum()
    n_not_w = ipw[~follow].sum()
    if n_follow_w == 0 or n_not_w == 0:
        return nan

    p_follow_adj = np.sum(ipw[follow] * pcr[follow]) / n_follow_w
    p_not_adj = np.sum(ipw[~follow] * pcr[~follow]) / n_not_w
    coverage_adj = n_follow_w / (n_follow_w + n_not_w)
    rrd_adj = p_follow_adj - p_not_adj
    cau_adj = rrd_adj * coverage_adj
    return {
        "CAU_adj": float(cau_adj), "CAU_pp_adj": float(cau_adj * 100.0),
        "RRD_adj": float(rrd_adj), "RRD_pp_adj": float(rrd_adj * 100.0),
        "Recovery_Ratio_adj": float(p_follow_adj / p_not_adj) if p_not_adj > 0 else np.nan,
        "IPW_ess_all": float(ipw.sum() ** 2 / np.sum(ipw ** 2)),
    }


# ============================================================
# 7. E-value (sensitivity to UNMEASURED confounding)
# ============================================================

def compute_e_value(rr: float) -> dict:
    """
    VanderWeele & Ding (2017) E-value for a risk ratio `rr`:
        E = RR + sqrt(RR * (RR - 1))  for RR >= 1 (RR < 1 is inverted first).
    """
    if rr is None or not np.isfinite(rr) or rr <= 0:
        return {"input_rr": rr, "rr_used": np.nan, "e_value": np.nan}
    rr_used = (1.0 / rr) if rr < 1 else rr
    e_value = 1.0 if rr_used == 1.0 else rr_used + np.sqrt(rr_used * (rr_used - 1.0))
    return {"input_rr": float(rr), "rr_used": float(rr_used), "e_value": float(e_value)}


def compute_e_value_for_ci(rr: float, ci_lo: float, ci_hi: float) -> dict:
    """E-value for the point estimate and for the CI bound closest to the null (RR = 1)."""
    point = compute_e_value(rr)
    if not (np.isfinite(rr) and np.isfinite(ci_lo) and np.isfinite(ci_hi)):
        return {**point, "ci_bound_used": np.nan, "e_value_ci": np.nan}
    bound = ci_lo if rr >= 1.0 else ci_hi
    if bound <= 0:
        return {**point, "ci_bound_used": float(bound), "e_value_ci": np.nan}
    if (rr >= 1.0 and bound <= 1.0) or (rr < 1.0 and bound >= 1.0):
        return {**point, "ci_bound_used": float(bound), "e_value_ci": 1.0}
    return {**point, "ci_bound_used": float(bound), "e_value_ci": compute_e_value(bound)["e_value"]}


def recovery_ratio_to_risk_ratio(recovery_ratio: float) -> float:
    """The Recovery Ratio (p_recovery_follow / p_recovery_not) is already a risk ratio on pCR."""
    return recovery_ratio


# ============================================================
# 8. Cross-dataset stability
# ============================================================

def compute_cross_dataset_stability(
    combined: pd.DataFrame,
    value_col: str = "DR_uplift",
    method_col: str = "Method",
    dataset_col: str = "DataSet",
    higher_is_better: bool = None,
) -> pd.DataFrame:
    """
    Per method, how stable `value_col` is across datasets: mean, std,
    coefficient of variation (CV = std / |mean|), sign consistency (share of
    datasets with the same sign as the mean) and the number of datasets
    where the method is favourable (value > 0 if `higher_is_better`, value
    < 0 otherwise). By default uplift metrics (DR_uplift, IPW_uplift) are
    "lower is better" and everything else (CAU_pp, RRD_pp, ...) "higher is
    better"; Recovery_Ratio is favourable when > 1.
    """
    if higher_is_better is None:
        higher_is_better = not value_col.endswith("uplift")
    threshold = 1.0 if value_col.startswith("Recovery_Ratio") else 0.0

    rows = []
    for method, g in combined.groupby(method_col):
        vals = pd.to_numeric(g[value_col], errors="coerce").dropna().values
        n = len(vals)
        if n == 0:
            continue
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        centred = vals - threshold
        mean_c = mean_val - threshold
        n_fav = int(np.sum(centred > 0)) if higher_is_better else int(np.sum(centred < 0))
        rows.append({
            "Method": method, "Metric": value_col, "N_Datasets": n,
            "Mean": mean_val, "Std": std_val,
            "CV": abs(std_val / mean_val) if mean_val != 0 else np.nan,
            "Sign_Consistency": float(np.mean(np.sign(centred) == np.sign(mean_c))) if mean_c != 0 else np.nan,
            "N_Favorable_Datasets": n_fav,
            "Pct_Favorable_Datasets": n_fav / n,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("CV", ascending=True, na_position="last").reset_index(drop=True)
    return out
