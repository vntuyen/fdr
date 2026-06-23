# -*- coding: utf-8 -*-
"""
statistical_validation.py
==========================
Statistical-validation 
"""

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

# ============================================================
# 1. Bootstrap confidence intervals
# ============================================================

def bootstrap_ci_recovery_metrics(
    df: pd.DataFrame,
    outcome_col: str,
    n_boot: int = N_BOOTSTRAP,
    ci_level: float = CI_LEVEL,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """
    Patient-level bootstrap percentile CIs for Recovery_Ratio, CAU, CAU_pp,
    RRD, RRD_pp, computed by resampling PATIENTS WITH REPLACEMENT from `df`
    (a single method's REC dataframe, pooling all repeated-CV runs if
    present) and recomputing the metric from scratch on each resample.

    This complements (does not replace) the existing across-RUN std already
    reported in evaluation.py's *_summary tables: across-run std captures
    sensitivity to the repeated-CV seed, while the bootstrap CI here
    captures sampling variability given the actual patient cohort size
    (important for the smaller datasets where N is the binding constraint).
    """
    rng = np.random.RandomState(seed)
    n = len(df)
    if n == 0:
        return {"ci_level": ci_level, "n_boot": n_boot, "n": 0}

    follow = df["FOLLOW_REC"].values.astype(bool)
    has_pcr = "resp.pCR" in df.columns
    pcr = df["resp.pCR"].values if has_pcr else None

    def _metric_from_mask(follow_mask, pcr_vals):
        n_follow = follow_mask.sum()
        n_not = (~follow_mask).sum()
        out = {}
        if has_pcr and n_follow > 0 and n_not > 0:
            rec_follow = pcr_vals[follow_mask].sum()
            rec_not = pcr_vals[~follow_mask].sum()
            p_follow = rec_follow / n_follow
            p_not = rec_not / n_not
            coverage = n_follow / (n_follow + n_not)
            rrd = p_follow - p_not
            cau = rrd * coverage
            out["Recovery_Ratio"] = (p_follow / p_not) if p_not > 0 else np.nan
            out["RRD"] = rrd
            out["RRD_pp"] = rrd * 100.0
            out["CAU"] = cau
            out["CAU_pp"] = cau * 100.0
        else:
            out.update({"Recovery_Ratio": np.nan, "RRD": np.nan, "RRD_pp": np.nan,
                        "CAU": np.nan, "CAU_pp": np.nan})
        return out

    boot_metrics = {k: [] for k in ["Recovery_Ratio", "RRD", "RRD_pp", "CAU", "CAU_pp"]}
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        m = _metric_from_mask(follow[idx], pcr[idx] if has_pcr else None)
        for k in boot_metrics:
            boot_metrics[k].append(m[k])

    alpha = 1 - ci_level
    result = {"ci_level": ci_level, "n_boot": n_boot, "n": n}
    for k, vals in boot_metrics.items():
        arr = np.asarray(vals, dtype=float)
        finite = arr[np.isfinite(arr)]
        if len(finite) == 0:
            result[f"{k}_ci_lo"] = np.nan
            result[f"{k}_ci_hi"] = np.nan
            result[f"{k}_boot_mean"] = np.nan
        else:
            result[f"{k}_ci_lo"] = float(np.percentile(finite, 100 * alpha / 2))
            result[f"{k}_ci_hi"] = float(np.percentile(finite, 100 * (1 - alpha / 2)))
            result[f"{k}_boot_mean"] = float(np.mean(finite))
        result[f"{k}_n_valid_boot"] = int(len(finite))
    return result


def bootstrap_ci_dr_uplift(
    df: pd.DataFrame,
    treatment_plans: list,
    outcome_col: str,
    n_boot: int = N_BOOTSTRAP,
    ci_level: float = CI_LEVEL,
    seed: int = BOOTSTRAP_SEED,
) -> dict:

    rng = np.random.RandomState(seed)
    n = len(df)
    if n == 0 or not all(f"PROP_{tp}" in df.columns for tp in treatment_plans):
        return {"ci_level": ci_level, "n_boot": n_boot, "n": n, "DR_uplift_ci_lo": np.nan, "DR_uplift_ci_hi": np.nan}

    propensity_full = df[[f"PROP_{tp}" for tp in treatment_plans]].values

    boot_vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        resample = df.iloc[idx].reset_index(drop=True)
        prop_resample = propensity_full[idx]
        try:
            pv = policy_value.evaluate_policy_value(resample, treatment_plans, outcome_col, prop_resample)
            boot_vals.append(pv["DR_uplift"])
        except Exception:
            continue

    alpha = 1 - ci_level
    arr = np.asarray(boot_vals, dtype=float)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return {"ci_level": ci_level, "n_boot": n_boot, "n": n,
                "DR_uplift_ci_lo": np.nan, "DR_uplift_ci_hi": np.nan, "DR_uplift_n_valid_boot": 0}
    return {
        "ci_level": ci_level, "n_boot": n_boot, "n": n,
        "DR_uplift_ci_lo": float(np.percentile(finite, 100 * alpha / 2)),
        "DR_uplift_ci_hi": float(np.percentile(finite, 100 * (1 - alpha / 2))),
        "DR_uplift_boot_mean": float(np.mean(finite)),
        "DR_uplift_n_valid_boot": int(len(finite)),
    }

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
    Cross-dataset bootstrap CI: for each method, resamples DATASETS (not
    patients) with replacement and recomputes the across-dataset mean of
    `value_col` on each resample, to get a 95% percentile CI on "the mean
    of this metric across all datasets" for that method.

    This is the correct resampling UNIT for a cross-dataset summary: each
    dataset already has its own internal patient-level bootstrap CI
    (bootstrap_ci_recovery_metrics) and its own across-run mean/std (the
    repeated stratified-CV resamples, see config.REPEAT_SEEDS /
    run_experiments.py), so the remaining source of uncertainty this
    function targets is "how much would the across-dataset average change
    if we had drawn a different set of datasets" -- the dataset, not the
    patient or the run, is the unit being resampled.

    `per_dataset_values` should have one row per (dataset, method) -- e.g.
    the already-run-averaged `combined` table built in
    evaluation.evaluate_all_datasets, where `value_col` is the per-dataset,
    per-method mean across the repeated stratified resamples (CAU_pp,
    RRD_pp, Recovery_Ratio, ...).

    Returns one row per method: N_Datasets, the plain across-dataset mean
    and std, and the 95% bootstrap CI on that mean.
    """
    rng = np.random.RandomState(seed)
    rows = []

    for method, g in per_dataset_values.groupby(method_col):
        vals = g[value_col].dropna().values
        n = len(vals)
        if n == 0:
            continue

        point_mean = float(np.mean(vals))
        point_std = float(np.std(vals, ddof=1)) if n > 1 else 0.0

        boot_means = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.randint(0, n, size=n)
            boot_means[i] = np.mean(vals[idx])

        alpha = 1 - ci_level
        rows.append({
            "Method": method,
            "N_Datasets": n,
            f"{value_col}_mean": point_mean,
            f"{value_col}_std": point_std,
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

def permutation_test_recovery_metrics(
    df: pd.DataFrame,
    outcome_col: str,
    n_perm: int = N_PERMUTATIONS,
    seed: int = BOOTSTRAP_SEED,
    metric: str = "CAU_pp",
) -> dict:
    """
    Permutation test against the null "the recommendation carries no
    information about who recovers / what RCB a patient gets". Builds the
    null distribution by permuting CURRENT_TP (the arm actually received)
    independently of REC_TP and outcome -- this breaks the
    recommendation-outcome relationship while exactly preserving the
    marginal arm-assignment proportions and the marginal outcome
    distribution, which is the relevant null for "is FOLLOW_REC informative"
    rather than a null about treatment assignment itself.

    Returns the two-sided permutation p-value for the observed `metric`
    against its null distribution, plus the null distribution's mean/std for
    reference.
    """
    rng = np.random.RandomState(seed)
    n = len(df)
    if n == 0 or "CURRENT_TP" not in df.columns or "REC_TP" not in df.columns:
        return {"metric": metric, "p_value": np.nan, "n_perm": n_perm, "n": n}

    rec_tp = df["REC_TP"].values
    current_tp = df["CURRENT_TP"].values
    has_pcr = "resp.pCR" in df.columns
    pcr = df["resp.pCR"].values if has_pcr else None

    def _metric_value(follow_mask):
        n_follow = follow_mask.sum()
        n_not = (~follow_mask).sum()
        if n_follow == 0 or n_not == 0 or not has_pcr:
            return np.nan
        p_follow = pcr[follow_mask].sum() / n_follow
        p_not = pcr[~follow_mask].sum() / n_not
        coverage = n_follow / (n_follow + n_not)
        rrd = p_follow - p_not
        cau = rrd * coverage
        if metric == "CAU":
            return cau
        if metric == "CAU_pp":
            return cau * 100.0
        if metric == "RRD":
            return rrd
        if metric == "RRD_pp":
            return rrd * 100.0
        if metric == "Recovery_Ratio":
            return (p_follow / p_not) if p_not > 0 else np.nan
        return np.nan

    observed_follow = (rec_tp == current_tp)
    observed = _metric_value(observed_follow)
    if not np.isfinite(observed):
        return {"metric": metric, "p_value": np.nan, "n_perm": n_perm, "n": n, "observed": observed}

    null_vals = np.empty(n_perm)
    for i in range(n_perm):
        permuted_current = rng.permutation(current_tp)
        follow_perm = (rec_tp == permuted_current)
        null_vals[i] = _metric_value(follow_perm)

    finite_null = null_vals[np.isfinite(null_vals)]
    if len(finite_null) == 0:
        return {"metric": metric, "p_value": np.nan, "n_perm": n_perm, "n": n, "observed": observed}

    # Two-sided permutation p-value: proportion of |null| >= |observed|,
    # with the standard +1 continuity correction.
    p_value = (np.sum(np.abs(finite_null) >= np.abs(observed)) + 1) / (len(finite_null) + 1)

    return {
        "metric": metric,
        "observed": float(observed),
        "null_mean": float(np.mean(finite_null)),
        "null_std": float(np.std(finite_null)),
        "p_value": float(p_value),
        "n_perm": n_perm,
        "n_valid_perm": int(len(finite_null)),
        "n": n,
    }



def evaluate_policy_value(
    df: pd.DataFrame,
    treatment_plans: list,
    outcome_col: str,
    propensity: np.ndarray,
    rec_col: str = "REC_TP",
    current_col: str = "CURRENT_TP",
) -> dict:
    """
    High-level entry point: given a REC-style dataframe (one row per patient,
    containing the observed outcome, the actually-received arm, the policy's
    recommended arm, and per-arm outcome-model predictions for every arm in
    `treatment_plans`), compute IPW and DR policy value plus the DR uplift
    over standard of care.

    `df` must contain one column per treatment plan with the outcome model's
    predicted outcome under that arm (this is `predicted_outcomes[tp]`,
    already written into every REC file by run_experiments.py).
    """
    y = df[outcome_col].astype(float).values
    actual_idx = df[current_col].map({tp: i for i, tp in enumerate(treatment_plans)}).values
    policy_idx = df[rec_col].map({tp: i for i, tp in enumerate(treatment_plans)}).values

    missing_q = [tp for tp in treatment_plans if tp not in df.columns]
    if missing_q:
        raise ValueError(
            f"evaluate_policy_value: missing per-arm outcome predictions for {missing_q}; "
            "REC file must include one column per treatment plan with the model's "
            "predicted outcome under that arm."
        )
    q_hat = df[treatment_plans].astype(float).values

    ipw = compute_ipw_value(y, actual_idx, propensity, policy_idx)
    dr = compute_dr_value(y, actual_idx, propensity, policy_idx, q_hat)
    soc_value = compute_standard_of_care_value(y)

    return {
        "IPW_value":        ipw["value"],
        "IPW_ess":          ipw["ess"],
        "IPW_n_matched":    ipw["n_matched"],
        "DR_value":         dr["value"],
        "DR_ess":           dr["ess"],
        "DR_n_matched":     dr["n_matched"],
        "SoC_value":        soc_value,
        "DR_uplift":        dr["value"] - soc_value,     # negative = improvement (lower RCB)
        "IPW_uplift":       ipw["value"] - soc_value,
        "N":                len(df),
    }


def permutation_test_dr_uplift(
    df: pd.DataFrame,
    treatment_plans: list,
    outcome_col: str,
    n_perm: int = N_PERMUTATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """
    Permutation test for DR_uplift against the null "the policy's chosen
    arm carries no information about outcome", built by permuting which
    patients are recorded as having received each arm (CURRENT_TP),
    independently of REC_TP, the propensity scores, and the per-arm Q
    predictions. This is a stronger and more relevant null than reshuffling
    Y directly, because it leaves the propensity/outcome MODELS untouched
    and only breaks the actual-arm <-> recommended-arm correspondence the DR
    estimator is built around.
    """
    rng = np.random.RandomState(seed)
    n = len(df)
    if n == 0 or not all(f"PROP_{tp}" in df.columns for tp in treatment_plans):
        return {"metric": "DR_uplift", "p_value": np.nan, "n_perm": n_perm, "n": n}

    propensity = df[[f"PROP_{tp}" for tp in treatment_plans]].values

    try:
        observed_pv = evaluate_policy_value(df, treatment_plans, outcome_col, propensity)
        observed = observed_pv["DR_uplift"]
    except Exception:
        return {"metric": "DR_uplift", "p_value": np.nan, "n_perm": n_perm, "n": n}

    current_tp = df["CURRENT_TP"].values
    null_vals = np.empty(n_perm)
    for i in range(n_perm):
        permuted = df.copy()
        permuted["CURRENT_TP"] = rng.permutation(current_tp)
        try:
            pv = evaluate_policy_value(permuted, treatment_plans, outcome_col, propensity)
            null_vals[i] = pv["DR_uplift"]
        except Exception:
            null_vals[i] = np.nan

    finite_null = null_vals[np.isfinite(null_vals)]
    if len(finite_null) == 0 or not np.isfinite(observed):
        return {"metric": "DR_uplift", "observed": observed, "p_value": np.nan, "n_perm": n_perm, "n": n}

    p_value = (np.sum(np.abs(finite_null) >= np.abs(observed)) + 1) / (len(finite_null) + 1)
    return {
        "metric": "DR_uplift",
        "observed": float(observed),
        "null_mean": float(np.mean(finite_null)),
        "null_std": float(np.std(finite_null)),
        "p_value": float(p_value),
        "n_perm": n_perm,
        "n_valid_perm": int(len(finite_null)),
        "n": n,
    }


# ============================================================
# 3. Multiplicity correction
# ============================================================

def apply_multiplicity_correction(p_values_df: pd.DataFrame, p_col: str = "p_value", alpha: float = 0.05) -> pd.DataFrame:
    """
    Adds Bonferroni and Benjamini-Hochberg (BH) FDR-corrected significance
    flags to a long-format table of p-values (one row per (dataset, method,
    metric) comparison, as produced by run_statistical_validation below).

    Both corrections are reported side by side: Bonferroni is the
    conservative family-wise-error-rate control, BH is the less conservative
    false-discovery-rate control -- showing both lets the paper report
    "how many of our headline comparisons survive a strict correction" while
    being transparent that a milder, still-standard correction is also
    informative given the number of comparisons made.
    """
    out = p_values_df.copy()
    m = out[p_col].notna().sum()
    if m == 0:
        out["bonferroni_p"] = np.nan
        out["bonferroni_significant"] = False
        out["bh_p"] = np.nan
        out["bh_significant"] = False
        return out

    out["bonferroni_p"] = (out[p_col] * m).clip(upper=1.0)
    out["bonferroni_significant"] = out["bonferroni_p"] < alpha

    # Benjamini-Hochberg FDR correction.
    valid = out[p_col].notna()
    ranks = out.loc[valid, p_col].rank(method="first")
    bh_p = out.loc[valid, p_col] * m / ranks
    # Enforce monotonicity (BH-adjusted p-values must be non-decreasing as
    # the raw p-value decreases) and cap at 1.0.
    sorted_idx = out.loc[valid].sort_values(p_col, ascending=False).index
    running_min = np.inf
    bh_p_monotone = {}
    for idx in sorted_idx:
        running_min = min(running_min, bh_p.loc[idx])
        bh_p_monotone[idx] = min(running_min, 1.0)
    out["bh_p"] = np.nan
    for idx, val in bh_p_monotone.items():
        out.loc[idx, "bh_p"] = val
    out["bh_significant"] = out["bh_p"] < alpha

    return out


# ============================================================
# 4. Covariate balance / Standardized Mean Difference (SMD)
# ============================================================

def compute_smd_table(X: pd.DataFrame, treatment_plans: list, weights: np.ndarray = None) -> pd.DataFrame:
    """
    Standardized Mean Difference (SMD) for every covariate (every column of
    X not in treatment_plans), comparing each treatment arm's distribution
    against the pooled distribution of all OTHER arms -- the standard
    multi-arm generalisation of the two-arm SMD diagnostic for "how
    confounded does treatment assignment look on the covariates we have".

    SMD = (mean_arm - mean_rest) / pooled_std

    Conventionally, |SMD| > 0.1 is treated as meaningful imbalance and
    |SMD| > 0.25 as substantial imbalance (Austin, 2009). If `weights` (e.g.
    inverse-propensity weights) is supplied, the SMD is computed on the
    WEIGHTED means/variances -- comparing the unweighted and weighted SMD
    tables for the same dataset is the standard way to show that propensity
    weighting actually improved covariate balance.
    """
    covariate_cols = [c for c in X.columns if c not in treatment_plans]
    rows = []

    for tp in treatment_plans:
        if tp not in X.columns:
            continue
        in_arm = X[tp].values == 1
        out_arm = ~in_arm
        if in_arm.sum() == 0 or out_arm.sum() == 0:
            continue

        w = weights if weights is not None else np.ones(len(X))

        for col in covariate_cols:
            vals = X[col].values.astype(float)
            w_in, w_out = w[in_arm], w[out_arm]
            v_in, v_out = vals[in_arm], vals[out_arm]

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
    """
    Per-arm summary of the SMD table: number and proportion of covariates
    with |SMD| > 0.1 ("meaningful imbalance") and > 0.25 ("substantial
    imbalance"), plus the max |SMD| observed for that arm -- a compact way
    to report "how confounded does this dataset's treatment assignment look"
    without reproducing the full per-covariate table in the main text.
    """
    if smd_df.empty:
        return pd.DataFrame(columns=["Arm", "N_Covariates", "N_SMD_gt_0.1", "Pct_SMD_gt_0.1",
                                      "N_SMD_gt_0.25", "Pct_SMD_gt_0.25", "Max_Abs_SMD"])

    rows = []
    for arm, g in smd_df.groupby("Arm"):
        n_cov = g["Abs_SMD"].notna().sum()
        n_gt_01 = (g["Abs_SMD"] > 0.1).sum()
        n_gt_025 = (g["Abs_SMD"] > 0.25).sum()
        rows.append({
            "Arm": arm,
            "N_Covariates": int(n_cov),
            "N_SMD_gt_0.1": int(n_gt_01),
            "Pct_SMD_gt_0.1": float(n_gt_01 / n_cov) if n_cov > 0 else np.nan,
            "N_SMD_gt_0.25": int(n_gt_025),
            "Pct_SMD_gt_0.25": float(n_gt_025 / n_cov) if n_cov > 0 else np.nan,
            "Max_Abs_SMD": float(g["Abs_SMD"].max()) if n_cov > 0 else np.nan,
        })
    return pd.DataFrame(rows)


# ============================================================
# 5. IPW-adjusted CAU (sensitivity check against the naive CAU)
# ============================================================

def compute_adjusted_cau(
    df: pd.DataFrame,
    treatment_plans: list,
    outcome_col: str,
    clip: float = PROPENSITY_CLIP,
) -> dict:
    """
    IPW-adjusted version of CAU/RRD/Recovery-Ratio: instead of raw
    (Following_Count, NotFollowing_Count, Following_Recovery,
    NotFollowing_Recovery) subgroup counts, uses INVERSE-PROPENSITY-WEIGHTED
    pseudo-counts, i.e. each patient contributes 1/e_{actual_arm}(x) instead
    of 1 to their (FOLLOW_REC) subgroup. This reweights the naive comparison
    to better approximate the covariate distribution under random
    assignment, WITHOUT changing the underlying statistic's definition --
    making it a direct, like-for-like sensitivity check against the
    unadjusted CAU already reported elsewhere (large discrepancies between
    adjusted and unadjusted CAU signal that confounding was doing a lot of
    work in the naive number).

    Requires PROP_<tp> propensity columns (written by run_experiments.py).
    """
    if not all(f"PROP_{tp}" in df.columns for tp in treatment_plans):
        return {"CAU_adj": np.nan, "RRD_adj": np.nan, "Recovery_Ratio_adj": np.nan}

    follow = df["FOLLOW_REC"].values.astype(bool)
    current_tp = df["CURRENT_TP"].values
    tp_to_idx = {tp: i for i, tp in enumerate(treatment_plans)}
    actual_idx = np.array([tp_to_idx.get(tp, -1) for tp in current_tp])
    propensity = df[[f"PROP_{tp}" for tp in treatment_plans]].values
    valid = actual_idx >= 0
    e_actual = np.full(len(df), np.nan)
    e_actual[valid] = propensity[valid, actual_idx[valid]]
    e_actual = np.clip(e_actual, clip, 1.0)
    ipw = 1.0 / e_actual  # IPW pseudo-weight per patient

    has_pcr = "resp.pCR" in df.columns
    if not has_pcr:
        return {"CAU_adj": np.nan, "RRD_adj": np.nan, "Recovery_Ratio_adj": np.nan}
    pcr = df["resp.pCR"].astype(float).values

    w_follow = ipw[follow]
    w_not = ipw[~follow]
    pcr_follow = pcr[follow]
    pcr_not = pcr[~follow]

    n_follow_w = w_follow.sum()
    n_not_w = w_not.sum()
    if n_follow_w == 0 or n_not_w == 0:
        return {"CAU_adj": np.nan, "RRD_adj": np.nan, "Recovery_Ratio_adj": np.nan}

    p_follow_adj = np.sum(w_follow * pcr_follow) / n_follow_w
    p_not_adj = np.sum(w_not * pcr_not) / n_not_w
    coverage_adj = n_follow_w / (n_follow_w + n_not_w)

    rrd_adj = p_follow_adj - p_not_adj
    cau_adj = rrd_adj * coverage_adj
    rr_adj = (p_follow_adj / p_not_adj) if p_not_adj > 0 else np.nan

    return {
        "CAU_adj": float(cau_adj), "CAU_pp_adj": float(cau_adj * 100.0),
        "RRD_adj": float(rrd_adj), "RRD_pp_adj": float(rrd_adj * 100.0),
        "Recovery_Ratio_adj": float(rr_adj),
    }


# ============================================================
# 6. E-value (sensitivity to UNMEASURED confounding)
# ============================================================

def compute_e_value(rr: float) -> dict:
    """
    VanderWeele & Ding (2017) E-value: the minimum strength of association
    (on the risk-ratio scale) that an unmeasured confounder would need to
    have with BOTH the policy's recommendation and the outcome to fully
    explain away an observed risk ratio `rr`, conditional on the measured
    covariates already adjusted for. This directly answers "how robust is
    our causal claim to confounders we didn't measure", which IPW/DR alone
    cannot answer (they only correct for confounders that were measured).

    `rr` should be a risk ratio >= 1 conceptually; if rr < 1 the function
    inverts it (1/rr) first, since the E-value is defined for the
    risk-ratio-away-from-1 in either direction.

    E-value formula (point estimate): for RR >= 1,
        E = RR + sqrt(RR * (RR - 1))
    For RR < 1, first invert: RR' = 1/RR, then apply the same formula to RR'.
    """
    if not np.isfinite(rr) or rr <= 0:
        return {"input_rr": rr, "rr_used": np.nan, "e_value": np.nan}

    rr_used = (1.0 / rr) if rr < 1 else rr
    if rr_used == 1.0:
        e_value = 1.0
    else:
        e_value = rr_used + np.sqrt(rr_used * (rr_used - 1.0))

    return {"input_rr": float(rr), "rr_used": float(rr_used), "e_value": float(e_value)}


def compute_e_value_for_ci(rr: float, ci_lo: float, ci_hi: float) -> dict:
    """
    E-value for the point estimate AND for the confidence-interval bound
    closest to the null (RR = 1) -- the latter is the more conservative,
    and more commonly reported, E-value: it answers "how strong would an
    unmeasured confounder need to be to move the CI bound closest to no-
    effect all the way to the null", which is a stronger and more honest
    sensitivity statement than the point-estimate E-value alone.
    """
    point = compute_e_value(rr)
    if not np.isfinite(ci_lo) or not np.isfinite(ci_hi):
        return {**point, "ci_bound_used": np.nan, "e_value_ci": np.nan}

    # The CI bound closest to the null (RR=1) is whichever of (ci_lo, ci_hi)
    # is nearer to 1 -- this is the bound that, if it crossed 1, would make
    # the result non-significant, so it is the relevant bound for the
    # "how robust is significance" sensitivity question.
    if (rr if np.isfinite(rr) else 1.0) >= 1.0:
        bound = ci_lo
    else:
        bound = ci_hi

    if not np.isfinite(bound) or bound <= 0:
        return {**point, "ci_bound_used": bound, "e_value_ci": np.nan}

    # If the CI already crosses the null, the "explain away the CI" E-value
    # is trivially 1 (no unmeasured confounding needed -- it's already not
    # significant).
    if (rr >= 1.0 and bound <= 1.0) or (rr < 1.0 and bound >= 1.0):
        return {**point, "ci_bound_used": float(bound), "e_value_ci": 1.0}

    ci_e = compute_e_value(bound)
    return {**point, "ci_bound_used": float(bound), "e_value_ci": ci_e["e_value"]}


def recovery_ratio_to_risk_ratio(recovery_ratio: float) -> float:
    """
    The Recovery Ratio statistic already used throughout this codebase
    (p_recovery_following / p_recovery_not_following) IS a risk ratio on the
    pCR outcome, so this is the natural quantity to feed to the E-value --
    no additional transformation is needed beyond clipping to a sane range.
    Provided as a small named wrapper purely for readability at call sites.
    """
    return recovery_ratio


# ============================================================
# 7. Cross-dataset stability metric (Comment 3)
# ============================================================

def compute_cross_dataset_stability(
    combined: pd.DataFrame,
    value_col: str = "DR_uplift",
    method_col: str = "Method",
    dataset_col: str = "DataSet",
) -> pd.DataFrame:
    """
    For every method, summarises how STABLE its `value_col` (typically
    DR_uplift, lower/more negative = better) is across datasets:

      - mean, std, coefficient of variation (CV = std / |mean|) across
        datasets -- a high CV means the average effect is not representative
        of any individual dataset.
      - sign-consistency: the fraction of datasets where the metric has the
        SAME SIGN as the overall mean -- a method that "wins on average" by
        being very good on one dataset and bad on others will have low
        sign-consistency even with a good mean.
      - n_datasets, n_favorable (datasets where the policy beats standard of
        care, i.e. value_col < 0 for DR_uplift) for transparency.

    This directly supports the "Cross-dataset variability and failure
    cases" subsection: a method/metric with high CV and low sign-consistency
    is flagged as unstable rather than reported as a single pooled average.
    """
    rows = []
    for method, g in combined.groupby(method_col):
        vals = g[value_col].dropna().values
        n = len(vals)
        if n == 0:
            continue
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        cv = abs(std_val / mean_val) if mean_val != 0 else np.nan
        sign_consistency = (
            float(np.mean(np.sign(vals) == np.sign(mean_val))) if mean_val != 0 else np.nan
        )
        n_favorable = int(np.sum(vals < 0))  # DR_uplift < 0 == improves on standard of care
        rows.append({
            "Method": method,
            "N_Datasets": n,
            "Mean": mean_val,
            "Std": std_val,
            "CV": cv,
            "Sign_Consistency": sign_consistency,
            "N_Favorable_Datasets": n_favorable,
            "Pct_Favorable_Datasets": n_favorable / n,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("CV", ascending=True, na_position="last").reset_index(drop=True)
    return out

