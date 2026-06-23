
"""
evaluation.py
=============
All evaluation metrics and reporting for FDR vs. baselines, across both the
CV and OOD scenarios:

  - Recovery Ratio (RR)       : (follow-recommendation recovery rate) /
                                 (not-follow recovery rate)
  - RRD / RRD_pp              : Recovery Rate Difference = p_follow - p_not
                                 (CAU without the coverage adjustment)
  - CAU / CAU_pp              : RRD * coverage (coverage = #follow / N) --
                                 the primary, coverage-adjusted uplift metric
  - extra_recoveries          : CAU * N (expected extra recoveries)
  - Average categorical RCB   : mean RCB score for patients who followed vs.
                                 did not follow the recommendation, per dataset

Metrics are computed per (method, run) from each method's REC file, then
aggregated as mean +/- std across the repeated runs (see run_experiments.py
for how runs/seeds are produced). Plots and pivot tables are written per
dataset and combined across all datasets.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import config
import statistical_validation as sv


OUTPUT_FOLDER_NAME = "output"


# ============================================================
# I/O helpers
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_dataset(path, encoding="ISO-8859-1"):
    return pd.read_csv(path, encoding=encoding, engine="python")


def _safe_max(arr, default=0.0):
    """Return max of array ignoring NaN; fall back to default if all NaN."""
    vals = np.asarray(arr, dtype=float)
    finite = vals[np.isfinite(vals)]
    return float(finite.max()) if len(finite) > 0 else default


def _safe_float(v, default=0.0):
    """Return float(v) if finite, else default."""
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def build_formatted(mean_df, std_df, decimals=2):
    """
    Element-wise "mean +/- std" string table for direct use in the paper.
    Falls back to a bare mean when std is unavailable/NaN (e.g. only one
    valid run), and to an empty cell when the mean itself is missing.
    """
    mean_arr = mean_df.values.astype(float)
    std_arr = std_df.values.astype(float) if std_df is not None else np.full_like(mean_arr, np.nan)
    out = np.empty(mean_arr.shape, dtype=object)
    for i in range(mean_arr.shape[0]):
        for j in range(mean_arr.shape[1]):
            m, s = mean_arr[i, j], std_arr[i, j]
            if pd.isna(m):
                out[i, j] = ""
            elif pd.isna(s):
                out[i, j] = f"{m:.{decimals}f}"
            else:
                out[i, j] = f"{m:.{decimals}f} \u00b1 {s:.{decimals}f}"
    return pd.DataFrame(out, index=mean_df.index, columns=mean_df.columns)


# ============================================================
# Summary statistics (per run)
# ============================================================

def save_summary_stats(summary_dict, method_name, run_id, df, outcome_col):
    """
    Accumulate per-(method, run) recovery and RCB statistics. Recovery stats
    are computed whenever resp.pCR exists in the dataframe (column presence
    is the gate, not a registry flag). `run_id` tags every row so that, after
    accumulating across all repeated runs, results can be aggregated per run
    first and then summarised (mean +/- std) across runs -- giving the final
    tables run-to-run variation to report, instead of a single point estimate.
    """
    if "resp.pCR" in df.columns:
        rec_stats = df.groupby("FOLLOW_REC")["resp.pCR"].agg(["count", "sum"]).reset_index()
        for _, row in rec_stats.iterrows():
            summary_dict["recovery"].append({
                "Method":     method_name,
                "Run":        run_id,
                "FOLLOW_REC": int(row["FOLLOW_REC"]),
                "Count":      int(row["count"]),
                "Recovery":   int(row["sum"]),
            })

    rcb_stats = df.groupby("FOLLOW_REC")[outcome_col].agg(["count", "mean", "std"]).reset_index()
    for _, row in rcb_stats.iterrows():
        summary_dict["rcb"].append({
            "Method":              method_name,
            "Run":                 run_id,
            "FOLLOW_REC":          int(row["FOLLOW_REC"]),
            "Count":               int(row["count"]),
            "Avg_categorical_RCB": round(row["mean"], 4),
            "Std_RCB":             round(row["std"], 4) if pd.notna(row["std"]) else np.nan,
        })


# ============================================================
# CAU / RR / RRD metrics
# ============================================================

def compute_cau(df):
    """
    Adds, per row (here: per Method x Run):
      - p_rec_follow, p_rec_not : recovery rate within each group
      - coverage                : #follow / N
      - RRD, RRD_pp             : Recovery Rate Difference = p_follow - p_not,
                                   i.e. CAU WITHOUT the coverage term (as if
                                   coverage = 1). Reported as a secondary
                                   metric for readers who do not want the
                                   #follow/N adjustment folded into the
                                   headline uplift number.
      - CAU, CAU_pp             : RRD * coverage (the primary, coverage-
                                   adjusted metric)
      - extra_recoveries        : CAU * N
    """
    df = df.copy()

    def safe_div(num, den):
        num = pd.to_numeric(num, errors="coerce")
        den = pd.to_numeric(den, errors="coerce")
        return np.where((den > 0) & np.isfinite(den), num / den, np.nan)

    df["p_rec_follow"] = safe_div(df["Following_Recovery"],    df["Following_Count"])
    df["p_rec_not"]    = safe_div(df["NotFollowing_Recovery"], df["NotFollowing_Count"])
    df["N"]            = df[["Following_Count", "NotFollowing_Count"]].sum(axis=1, min_count=1)
    df["coverage"]     = safe_div(df["Following_Count"], df["N"])

    df["RRD"]    = df["p_rec_follow"] - df["p_rec_not"]
    df["RRD_pp"] = df["RRD"] * 100.0

    df["CAU"]    = df["RRD"] * df["coverage"]
    df["CAU_pp"] = df["CAU"] * 100.0
    df["extra_recoveries"] = df["CAU"] * df["N"]
    return df



# ============================================================
# Core evaluation (run-aware)
# ============================================================

def _load_all_runs(output_path, data_name, method_name):
    """
    Loads the per-method prediction file for a dataset, preferring the
    multi-run file written by the repeated-CV driver
    (`<dataset>_<method>_REC_all_runs.csv`, with "Run"/"Seed" columns) and
    falling back to the single-run file (`<dataset>_<method>_REC.csv`,
    treated as a single Run=1 -- this is always the case for the OOD
    scenario, which has no repeated-run loop) if that's all that's available.
    Returns None if neither file exists.
    """
    all_runs_file = os.path.join(output_path, f"{data_name}_{method_name}_REC_all_runs.csv")
    legacy_file   = os.path.join(output_path, f"{data_name}_{method_name}_REC.csv")

    if os.path.exists(all_runs_file):
        df = load_dataset(all_runs_file)
        if "Run" not in df.columns:
            df["Run"] = 1
        return df

    if os.path.exists(legacy_file):
        df = load_dataset(legacy_file)
        df["Run"] = 1
        return df

    return None


def results_evaluation(data_name, output_path):
    """Compute Recovery Ratio / CAU / RRD, average-RCB summaries, and statistical
    validation (bootstrap CIs, permutation tests, IPW-adjusted CAU,
    E-values) for a single dataset, across whichever methods have REC files
    present."""
    cfg = config.DATASET_REGISTRY[data_name]
    outcome_col = cfg["outcome_col"]
    treatment_plans = cfg["treatment_plans"]
    methods = ["FDR"] + list(config.METHOD_META.keys())
    methods = list(dict.fromkeys(m for m in methods if m in config.METHOD_META))

    summary_stats = {"recovery": [], "rcb": []}
    statistical_validation_rows = []
    loaded_dfs = {}  # cache so the statistical-validation pass below doesn't re-read from disk

    for name in methods:
        df = _load_all_runs(output_path, data_name, name)
        if df is None:
            print(f"    [SKIP] Missing REC file for '{name}'")
            continue
        loaded_dfs[name] = df
        for run_id, run_df in df.groupby("Run"):
            save_summary_stats(summary_stats, name, int(run_id), run_df, outcome_col)


    result = {"output_dir": output_path}


    for name, df in loaded_dfs.items():
        n_runs_this_method = df["Run"].nunique() if "Run" in df.columns else 1

        row = {
            "DataSet": data_name, "Method": name,
            "N_Pooled_Across_Runs": len(df),
            "N_Runs": int(n_runs_this_method),
            "N_True_Cohort": len(df) / n_runs_this_method if n_runs_this_method else len(df),
        }

        boot = sv.bootstrap_ci_recovery_metrics(df, outcome_col)
        row.update({f"boot_{k}": v for k, v in boot.items()})

        perm_cau = sv.permutation_test_recovery_metrics(df, outcome_col, metric="CAU_pp")
        row["CAU_pp_perm_p"] = perm_cau.get("p_value", np.nan)
        row["CAU_pp_perm_observed"] = perm_cau.get("observed", np.nan)

        adj = sv.compute_adjusted_cau(df, treatment_plans, outcome_col)
        row.update(adj)

        # E-value computed on the point Recovery Ratio (a risk ratio on the
        # pCR outcome) and on its bootstrap CI bound closest to the null,
        # using the SAME bootstrap draws already computed above rather than
        # a separate analytic CI.
        rr_point = boot.get("Recovery_Ratio_boot_mean", np.nan)
        rr_lo = boot.get("Recovery_Ratio_ci_lo", np.nan)
        rr_hi = boot.get("Recovery_Ratio_ci_hi", np.nan)
        evalue = sv.compute_e_value_for_ci(rr_point, rr_lo, rr_hi)
        row.update({f"Recovery_Ratio_{k}": v for k, v in evalue.items()})

        if all(f"PROP_{tp}" in df.columns for tp in treatment_plans):
            perm_dr = sv.permutation_test_dr_uplift(df, treatment_plans, outcome_col)
            row["DR_uplift_perm_p"] = perm_dr.get("p_value", np.nan)
            row["DR_uplift_perm_observed"] = perm_dr.get("observed", np.nan)
            dr_boot = sv.bootstrap_ci_dr_uplift(df, treatment_plans, outcome_col)
            row.update({f"boot_{k}": v for k, v in dr_boot.items()})

        statistical_validation_rows.append(row)

    if statistical_validation_rows:
        sv_df = pd.DataFrame(statistical_validation_rows)

        result["statistical_validation"] = sv_df
    else:
        print(f"  [INFO] No REC files available for '{data_name}' -- skipping statistical validation.")


    if loaded_dfs:
        any_df = next(iter(loaded_dfs.values()))
        covariate_cols = [c for c in any_df.columns
                          if c not in (treatment_plans + [outcome_col, "REC_TP", "CURRENT_TP",
                                                           "FOLLOW_REC", "Run", "Seed", "resp.pCR"])
                          and not c.startswith("PROP_")
                          and c not in treatment_plans]
        # Reconstruct one-hot TP columns from CURRENT_TP for the SMD
        # computation (REC files store CURRENT_TP as a string label, not the
        # original one-hot columns).
        smd_input = any_df[covariate_cols].select_dtypes(include=[np.number]).copy()
        for tp in treatment_plans:
            smd_input[tp] = (any_df["CURRENT_TP"] == tp).astype(int)

        try:
            smd_table = sv.compute_smd_table(smd_input, treatment_plans)
            smd_summary = sv.summarize_smd_table(smd_table)

            smd_summary_path = os.path.join(output_path, f"{data_name}_CovariateBalance_SMD_summary.csv")
            smd_summary.to_csv(smd_summary_path, index=False)
            print(f"  [SAVED] {smd_summary_path}")

            result["smd_table"] = smd_table
            result["smd_summary"] = smd_summary
        except Exception as e:
            print(f"  [WARN] Covariate-balance (SMD) computation failed for '{data_name}': {e}")


    # ---- Recovery summary: per-run CAU/RRD, then mean +/- std across runs ----
    if summary_stats["recovery"]:
        raw = pd.DataFrame(summary_stats["recovery"])

        pivot = raw.pivot_table(
            index=["Method", "Run"], columns="FOLLOW_REC",
            values=["Count", "Recovery"], aggfunc="sum", fill_value=0
        )
        pivot.columns = [
            "NotFollowing_Count", "Following_Count",
            "NotFollowing_Recovery", "Following_Recovery",
        ]
        pivot = pivot.reset_index()

        with np.errstate(divide="ignore", invalid="ignore"):
            follow_rate = np.where(pivot["Following_Count"] > 0,
                                    pivot["Following_Recovery"] / pivot["Following_Count"], np.nan)
            notfollow_rate = np.where(pivot["NotFollowing_Count"] > 0,
                                       pivot["NotFollowing_Recovery"] / pivot["NotFollowing_Count"], np.nan)
            pivot["Recovery_Ratio"] = np.where(
                notfollow_rate > 0, follow_rate / notfollow_rate, np.nan
            )

        pivot = compute_cau(pivot)
        # NOTE: per-run detail (one row per method per repeated-CV run) is
        # kept in memory only and no longer written as a separate
        # <dataset>_Recovery_Metrics_per_run.csv file.

        metric_cols = ["Recovery_Ratio", "CAU", "CAU_pp", "RRD", "RRD_pp", "extra_recoveries"]
        agg = pivot.groupby("Method")[metric_cols].agg(["mean", "std"])
        agg.columns = [col if stat == "mean" else f"{col}_std" for col, stat in agg.columns]
        agg = agg.reset_index()

        count_means = pivot.groupby("Method")[
            ["Following_Count", "NotFollowing_Count", "Following_Recovery", "NotFollowing_Recovery"]
        ].mean().reset_index()

        n_runs = pivot.groupby("Method")["Run"].nunique().reset_index().rename(columns={"Run": "N_Runs"})
        n_valid_cau = pivot.groupby("Method")["CAU"].apply(lambda s: s.notna().sum()) \
                            .reset_index().rename(columns={"CAU": "N_Valid_CAU_Runs"})

        agg = (agg.merge(count_means, on="Method", how="left")
                   .merge(n_runs,      on="Method", how="left")
                   .merge(n_valid_cau, on="Method", how="left"))

        # ---- Merge in 95% bootstrap CIs + permutation p-value:
        #      computed earlier in this function (statistical_validation_rows
        #      / sv_df) on the SAME pooled-across-runs patient data, so the
        #      headline CAU/RRD/Recovery-Ratio table reports a 95% CI next
        #      to every point estimate without needing a separate file. ----
        if statistical_validation_rows:
            ci_cols = {
                "Method": "Method",
                "boot_CAU_ci_lo": "CAU_ci95_lo", "boot_CAU_ci_hi": "CAU_ci95_hi",
                "boot_CAU_pp_ci_lo": "CAU_pp_ci95_lo", "boot_CAU_pp_ci_hi": "CAU_pp_ci95_hi",
                "boot_RRD_ci_lo": "RRD_ci95_lo", "boot_RRD_ci_hi": "RRD_ci95_hi",
                "boot_RRD_pp_ci_lo": "RRD_pp_ci95_lo", "boot_RRD_pp_ci_hi": "RRD_pp_ci95_hi",
                "boot_Recovery_Ratio_ci_lo": "Recovery_Ratio_ci95_lo",
                "boot_Recovery_Ratio_ci_hi": "Recovery_Ratio_ci95_hi",
                "CAU_pp_perm_p": "CAU_pp_perm_p",
            }
            ci_df = sv_df[[c for c in ci_cols if c in sv_df.columns]].rename(columns=ci_cols)
            agg = agg.merge(ci_df, on="Method", how="left")

        agg_path = os.path.join(output_path, f"{data_name}_Recovery_Metrics_summary.csv")
        agg.to_csv(agg_path, index=False)
        print(f"  [SAVED] {agg_path}  (CAU/RRD/Recovery-Ratio with 95% bootstrap CI)")

        generate_recovery_ratio_plot(agg, output_path, data_name)
        generate_recovery_plot(agg, output_path, data_name)

        result["recovery_summary"] = agg
    else:
        print(f"  [INFO] No resp.pCR column found in any REC file for '{data_name}' -- skipping recovery plots.")

    # ---- RCB summary: per-run subgroup means, then mean +/- std ACROSS RUNS ----
    if summary_stats["rcb"]:
        rcb_raw = pd.DataFrame(summary_stats["rcb"])

        _pv_cnt = rcb_raw.pivot_table(
            index=["Method", "Run"], columns="FOLLOW_REC",
            values="Count", aggfunc="mean", fill_value=0
        ).rename(columns={0: "NotFollowing_Count", 1: "Following_Count"})

        _pv_avg = rcb_raw.pivot_table(
            index=["Method", "Run"], columns="FOLLOW_REC",
            values="Avg_categorical_RCB", aggfunc="mean", fill_value=np.nan
        ).rename(columns={0: "NotFollowing_Avg_RCB", 1: "Following_Avg_RCB"})

        rcb_pivot_per_run = pd.concat([_pv_cnt, _pv_avg], axis=1).reset_index()

        rcb_per_run_path = os.path.join(output_path, f"{data_name}_RCB_Score_Comparison_per_run.csv")
        rcb_pivot_per_run.to_csv(rcb_per_run_path, index=False)

        # Mean +/- std ACROSS THE REPEATED RUNS (not within-run patient
        # dispersion) -- the run-to-run variation the table needs.
        rcb_summary = rcb_pivot_per_run.groupby("Method").agg(
            Following_Avg_RCB        = ("Following_Avg_RCB", "mean"),
            Following_Avg_RCB_std    = ("Following_Avg_RCB", "std"),
            NotFollowing_Avg_RCB     = ("NotFollowing_Avg_RCB", "mean"),
            NotFollowing_Avg_RCB_std = ("NotFollowing_Avg_RCB", "std"),
            Following_Count          = ("Following_Count", "mean"),
            NotFollowing_Count       = ("NotFollowing_Count", "mean"),
        ).reset_index()

        n_runs_rcb = rcb_pivot_per_run.groupby("Method")["Run"].nunique() \
                                       .reset_index().rename(columns={"Run": "N_Runs"})
        rcb_summary = rcb_summary.merge(n_runs_rcb, on="Method", how="left")

        rcb_summary.to_csv(os.path.join(output_path, f"{data_name}_RCB_Score_Comparison.csv"), index=False)
        generate_rcb_barplot(rcb_summary, output_path, data_name)
        result["rcb_summary"] = rcb_summary

    return result


# ============================================================
# Plotting helpers
# ============================================================

def _figure_size(data_name):
    n_methods = 1 + len(config.METHOD_META)  # FDR + baselines
    if n_methods > 7 or "3TS" in data_name:
        return (max(13, n_methods * 1.4), 8)
    return (9, 8)


def _sort_methods(df, col="Method"):
    """Sort df rows by ALL_METHOD_COL_ORDER, keeping only rows with known methods."""
    present = [m for m in config.ALL_METHOD_COL_ORDER if m in df[col].values]
    df = df.copy()
    df[col] = pd.Categorical(df[col], categories=present, ordered=True)
    return df.sort_values(col).dropna(subset=[col]).reset_index(drop=True)


def generate_recovery_ratio_plot(df, output_path, data_name):
    df = _sort_methods(df)
    df = df.dropna(subset=["Recovery_Ratio"]).reset_index(drop=True)
    if df.empty:
        print(f"  [SKIP] No valid Recovery_Ratio rows for {data_name}")
        return

    methods = df["Method"].tolist()
    ratios = df["Recovery_Ratio"].values.astype(float)
    has_std = "Recovery_Ratio_std" in df.columns
    stds = (pd.to_numeric(df["Recovery_Ratio_std"], errors="coerce").fillna(0.0).values
            if has_std else np.zeros(len(ratios)))
    x = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=_figure_size(data_name))
    bars = ax.bar(x, ratios, width=0.6, yerr=stds, capsize=4,
                   color=[config.MODELS_COLOUR.get(m, "gray") for m in methods],
                   edgecolor="black")

    for bar, ratio, s in zip(bars, ratios, stds):
        if np.isfinite(ratio):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + s + 0.02,
                    f"{ratio:.2f}", ha="center", fontsize=10)

    finite_ratios = ratios[np.isfinite(ratios)]
    extra = float(np.nanmax(stds)) if len(stds) else 0.0
    ymax = (float(finite_ratios.max()) if len(finite_ratios) > 0 else 1.0) + 0.5 + extra

    title_suffix = " (mean \u00b1 std across runs)" if has_std else ""
    ax.set_title(f"{config.DATASET_TITLES.get(data_name, data_name)} -- Recovery Ratio{title_suffix}", fontsize=15)
    ax.set_ylabel("Recovery Ratio", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=13)
    ax.axhline(1.0, linestyle="--", linewidth=1, color="black", alpha=0.6)
    ax.set_ylim(0, max(1.5, ymax))
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    path = os.path.join(output_path, f"{data_name}_Recovery_Ratio.png")
    plt.savefig(path)
    plt.close()
    print(f"  [PLOT] Saved -> {path}")


def generate_recovery_plot(df, output_path, data_name):
    df = _sort_methods(df)
    if df.empty:
        return

    methods = df["Method"].tolist()
    following = pd.to_numeric(df["Following_Recovery"], errors="coerce").fillna(0).values
    not_following = pd.to_numeric(df["NotFollowing_Recovery"], errors="coerce").fillna(0).values
    following_count = pd.to_numeric(df["Following_Count"], errors="coerce").fillna(0).values
    nfollowing_count = pd.to_numeric(df["NotFollowing_Count"], errors="coerce").fillna(0).values
    x = np.arange(len(methods))

    bar_width = 0.4
    gap = 0.02
    fig, ax = plt.subplots(figsize=_figure_size(data_name))

    bars1 = ax.bar(x - (bar_width / 2 + gap / 2), following, bar_width)
    bars2 = ax.bar(x + (bar_width / 2 + gap / 2), not_following, bar_width)

    for i, method in enumerate(methods):
        colour = config.MODELS_COLOUR.get(method, "gray")
        for bars, hatch in [(bars1, "+"), (bars2, "/")]:
            bars[i].set_facecolor(colour)
            bars[i].set_edgecolor("black")
            bars[i].set_hatch(hatch)

    for bar, rec, cnt in zip(bars1, following, following_count):
        h = _safe_float(bar.get_height())
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                f"{round(rec)}\ntotal={round(cnt)}", ha="center", fontsize=9)
    for bar, rec, cnt in zip(bars2, not_following, nfollowing_count):
        h = _safe_float(bar.get_height())
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                f"{round(rec)}\ntotal={round(cnt)}", ha="center", fontsize=9)

    legend_patches = [
        Patch(facecolor="gray", hatch="+", label="Following", edgecolor="black"),
        Patch(facecolor="gray", hatch="/", label="Not Following", edgecolor="black"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=13,
              handlelength=2.5, handleheight=1.8, framealpha=0.6)

    ax.set_title(config.DATASET_TITLES.get(data_name, data_name), fontsize=15)
    ax.set_ylabel("Number of Recovered Patients (mean across runs)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=13)

    all_vals = np.concatenate([following, not_following])
    ymax = _safe_max(all_vals, default=10.0) + 10
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    path = os.path.join(output_path, f"{data_name}_Recovery_comparison.png")
    plt.savefig(path)
    plt.close()
    print(f"  [PLOT] Saved -> {path}")


def generate_rcb_barplot(df, output_path, data_name):
    """
    Mean RCB score per subgroup (followed recommendation vs not), with error
    bars showing the standard error of the mean ACROSS THE REPEATED RUNS
    (SEM = across-run SD / sqrt(n_runs)), rather than within-run patient
    dispersion -- this is what gives the figure run-to-run variation to show.
    """
    df = _sort_methods(df)
    if df.empty:
        return

    methods = df["Method"].tolist()
    f_avg = pd.to_numeric(df["Following_Avg_RCB"], errors="coerce").values
    nf_avg = pd.to_numeric(df["NotFollowing_Avg_RCB"], errors="coerce").values
    f_std = pd.to_numeric(df.get("Following_Avg_RCB_std"), errors="coerce").values
    nf_std = pd.to_numeric(df.get("NotFollowing_Avg_RCB_std"), errors="coerce").values
    n_runs = pd.to_numeric(df.get("N_Runs"), errors="coerce").fillna(1).values
    f_cnt = pd.to_numeric(df.get("Following_Count"), errors="coerce").fillna(0).values
    nf_cnt = pd.to_numeric(df.get("NotFollowing_Count"), errors="coerce").fillna(0).values

    with np.errstate(divide="ignore", invalid="ignore"):
        f_sem = np.where(n_runs > 1, np.nan_to_num(f_std) / np.sqrt(np.where(n_runs > 1, n_runs, 1)), 0.0)
        nf_sem = np.where(n_runs > 1, np.nan_to_num(nf_std) / np.sqrt(np.where(n_runs > 1, n_runs, 1)), 0.0)

    x = np.arange(len(methods))
    bar_width = 0.4
    gap = 0.02
    fig, ax = plt.subplots(figsize=_figure_size(data_name))

    bars1 = ax.bar(x - (bar_width / 2 + gap / 2), np.nan_to_num(f_avg), bar_width,
                    yerr=f_sem, capsize=4, ecolor="black")
    bars2 = ax.bar(x + (bar_width / 2 + gap / 2), np.nan_to_num(nf_avg), bar_width,
                    yerr=nf_sem, capsize=4, ecolor="black")

    for i, method in enumerate(methods):
        colour = config.MODELS_COLOUR.get(method, "gray")
        for bars, hatch in [(bars1, "+"), (bars2, "/")]:
            bars[i].set_facecolor(colour)
            bars[i].set_edgecolor("black")
            bars[i].set_hatch(hatch)

    for bar, avg, sem, cnt in zip(bars1, f_avg, f_sem, f_cnt):
        if not np.isfinite(avg):
            continue
        h = _safe_float(bar.get_height())
        ax.text(bar.get_x() + bar.get_width() / 2, h + sem + 0.05,
                f"{avg:.2f}\nn={round(cnt)}", ha="center", fontsize=9)
    for bar, avg, sem, cnt in zip(bars2, nf_avg, nf_sem, nf_cnt):
        if not np.isfinite(avg):
            continue
        h = _safe_float(bar.get_height())
        ax.text(bar.get_x() + bar.get_width() / 2, h + sem + 0.05,
                f"{avg:.2f}\nn={round(cnt)}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=13)

    legend_patches = [
        Patch(facecolor="gray", hatch="+", label="Following", edgecolor="black"),
        Patch(facecolor="gray", hatch="/", label="Not Following", edgecolor="black"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=13,
              handlelength=2.0, handleheight=2.0, framealpha=0.6)

    ax.set_title(config.DATASET_TITLES.get(data_name, data_name), fontsize=15)
    ax.set_ylabel("Average RCB Score (mean \u00b1 SEM across runs)", fontsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    path = os.path.join(output_path, f"{data_name}_RCB_Score_comparison.png")
    plt.savefig(path)
    plt.close()
    print(f"  [PLOT] Saved -> {path}")





def generate_modality_comparison_plot(comp_df, output_folder, fname_stub, metric_label="CAU (pp)"):
    df = comp_df.dropna(subset=["Clinical_mean", "Multiomics_mean"], how="all").copy()
    if df.empty:
        return

    methods = df["Method"].astype(str).tolist()
    clin_mean = pd.to_numeric(df["Clinical_mean"], errors="coerce").values
    clin_std = pd.to_numeric(df["Clinical_std"], errors="coerce").fillna(0.0).values
    multi_mean = pd.to_numeric(df["Multiomics_mean"], errors="coerce").values
    multi_std = pd.to_numeric(df["Multiomics_std"], errors="coerce").fillna(0.0).values

    x = np.arange(len(methods))
    bar_width = 0.38

    fig, ax = plt.subplots(figsize=(max(12, len(methods) * 1.1), 7))
    ax.bar(x - bar_width / 2, np.nan_to_num(clin_mean), bar_width, yerr=clin_std, capsize=4,
           label="Clinical", color="lightsteelblue", edgecolor="black")
    ax.bar(x + bar_width / 2, np.nan_to_num(multi_mean), bar_width, yerr=multi_std, capsize=4,
           label="Multi-omics", color="darkorange", edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=12)
    ax.set_ylabel(f"{metric_label}, mean \u00b1 std across datasets", fontsize=13)
    ax.set_title(f"Clinical vs Multi-omics (paired: TransNEO, ARTemis) -- {metric_label}", fontsize=14)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    path = os.path.join(output_folder, f"{OUTPUT_FOLDER_NAME}_{fname_stub}_Clinical_vs_Multiomics.png")
    plt.savefig(path)
    plt.close()
    print(f"  [PLOT] Saved -> {path}")


def generate_method_ranking_plot(ranking_df, value_col, output_folder, fname_stub, metric_label="CAU (pp)"):
    df = ranking_df.copy()
    if df.empty:
        return

    methods = df["Method"].astype(str).tolist()
    means = pd.to_numeric(df[f"{value_col}_mean"], errors="coerce").values
    stds = pd.to_numeric(df[f"{value_col}_std"], errors="coerce").fillna(0.0).values
    colours = ["blue" if m == "FDR" else config.MODELS_COLOUR.get(m, "gray") for m in methods]
    edge_colours = ["red" if m == "FDR" else "black" for m in methods]
    linewidths = [2.5 if m == "FDR" else 1.0 for m in methods]

    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(max(12, len(methods) * 0.9), 7))
    bars = ax.bar(x, np.nan_to_num(means), yerr=stds, capsize=4,
                   color=colours, edgecolor=edge_colours, linewidth=linewidths)

    for bar, m, s in zip(bars, means, stds):
        if np.isfinite(m):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.3,
                    f"{m:.1f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=12)
    ax.set_ylabel(f"{metric_label}, mean \u00b1 std (multi-omics datasets)", fontsize=13)
    ax.set_title(f"Method comparison on multi-omics datasets only -- {metric_label}", fontsize=15)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    path = os.path.join(output_folder, f"{OUTPUT_FOLDER_NAME}_{fname_stub}_MultiomicsOnly_MethodRanking.png")
    plt.savefig(path)
    plt.close()
    print(f"  [PLOT] Saved -> {path}")


# ============================================================
# Clinical vs Multi-omics PAIRED comparison (same-cohort, same-patient) /
# broader multi-omics-only method ranking
# ============================================================

def build_paired_modality_comparison(combined, value_col, output_folder, fname_stub, decimals=2):
    """
    PAIRED Clinical-vs-Multi-omics comparison: for each method and each
    cohort in config.CLINICAL_MULTIOMICS_PAIRS (TransNEO, ARTemis), looks up
    `value_col` (a per-dataset, already run-averaged metric, e.g. "CAU_pp") for that cohort's clinical-only dataset AND its
    same-patient multi-omics dataset, and reports them side by side plus
    their difference.

    """
    rows = []
    for cohort, pair in config.CLINICAL_MULTIOMICS_PAIRS.items():
        clin_name, multi_name = pair["clinical"], pair["multiomics"]
        clin_vals = combined.loc[combined["DataSet"] == clin_name, ["Method", value_col]] \
                             .rename(columns={value_col: "Clinical_value"})
        multi_vals = combined.loc[combined["DataSet"] == multi_name, ["Method", value_col]] \
                              .rename(columns={value_col: "Multiomics_value"})
        merged = clin_vals.merge(multi_vals, on="Method", how="outer")
        merged.insert(0, "Cohort", cohort)
        rows.append(merged)

    if not rows:
        return pd.DataFrame(columns=["Method", "Clinical_mean", "Clinical_std", "Clinical_N_Datasets",
                                      "Multiomics_mean", "Multiomics_std", "Multiomics_N_Datasets",
                                      "Multiomics_minus_Clinical"])

    paired_long = pd.concat(rows, ignore_index=True)
    paired_long["Multiomics_minus_Clinical"] = paired_long["Multiomics_value"] - paired_long["Clinical_value"]

    paired_long_path = os.path.join(
        output_folder, f"{OUTPUT_FOLDER_NAME}_{fname_stub}_Clinical_vs_Multiomics_Paired_byCohort.csv"
    )
    order = [m for m in config.ALL_METHOD_COL_ORDER if m in paired_long["Method"].values]
    paired_long["Method"] = pd.Categorical(paired_long["Method"], categories=order, ordered=True)
    paired_long = paired_long.sort_values(["Method", "Cohort"]).reset_index(drop=True)
    paired_long.round(decimals).to_csv(paired_long_path, index=False)
    print(f"[SAVED] {paired_long_path}")

    # Across-cohort summary: mean +/- std OF THE PAIRED COHORTS (TransNEO,
    # ARTemis) only -- i.e. averaging the two paired observations per
    # method, not pooling raw patients across cohorts.
    summary = paired_long.groupby("Method").agg(
        Clinical_mean             = ("Clinical_value", "mean"),
        Clinical_std              = ("Clinical_value", "std"),
        Clinical_N_Datasets       = ("Clinical_value", "count"),
        Multiomics_mean           = ("Multiomics_value", "mean"),
        Multiomics_std            = ("Multiomics_value", "std"),
        Multiomics_N_Datasets     = ("Multiomics_value", "count"),
        Multiomics_minus_Clinical = ("Multiomics_minus_Clinical", "mean"),
        Multiomics_minus_Clinical_std = ("Multiomics_minus_Clinical", "std"),
    ).reset_index()
    summary["Method"] = pd.Categorical(summary["Method"], categories=order, ordered=True)
    summary = summary.sort_values("Method").reset_index(drop=True)

    path = os.path.join(output_folder, f"{OUTPUT_FOLDER_NAME}_{fname_stub}_Clinical_vs_Multiomics.csv")
    summary.round(decimals).to_csv(path, index=False)
    print(f"[SAVED] {path}")
    return summary


# ============================================================
# Clinical vs Multi-omics WITHIN COHORT figure: FDR vs. the 9-method mean
# ============================================================
#
# ONE figure, three panels -- CAU (pp), RRD (pp), Recovery Ratio. Within
# each panel the x-axis has FOUR clusters, in order:
#   TransNEO FDR | TransNEO All-method mean | ARTemis FDR | ARTemis All-method mean
# and within each cluster the LEFT bar is Clinical, the RIGHT bar is
# Multi-omics. Error bars are the across-RUN std for FDR (already computed
# in `combined`'s *_std columns) and the across-METHOD std for the
# all-method mean. Each bar
# is labelled with its value, and each cluster is annotated with
# Delta = Multi-omics - Clinical above the taller bar's error whisker.

_COHORT_PANEL_METRICS = [
    ("CAU_pp",         "CAU_pp_std",         "CAU (pp)"),
    ("RRD_pp",         "RRD_pp_std",         "RRD (pp)"),
    ("Recovery_Ratio", "Recovery_Ratio_std", "Recovery Ratio"),
]


def build_cohort_clinical_vs_multiomics_data(combined, decimals=4) -> pd.DataFrame:
    """
    Builds the long-format table the figure needs: one row per
    (Cohort in {"TransNEO", "ARTemis"}, Group in {"FDR", "All-method mean"},
    Modality in {"Clinical", "Multi-omics"}, Metric in {CAU_pp, RRD_pp,
    Recovery_Ratio}), with `mean` (the bar height) and `std` (the error bar).

    - "FDR" rows: FDR's own across-run mean/std for that dataset, taken
      directly from `combined` (already computed in results_evaluation as
      the mean/std of the per-run metric across the repeated CV runs).
    - "All-method mean" rows: mean of the per-method across-run means over
      every method in config.METHOD_META 
      with std taken ACROSS THOSE METHODS.
    """
    all_methods = list(config.METHOD_META.keys())  # 15 methods, FDR included
    rows = []

    for cohort, pair in config.CLINICAL_MULTIOMICS_PAIRS.items():
        modality_to_dataset = {"Clinical": pair["clinical"], "Multi-omics": pair["multiomics"]}

        for modality, dataset_name in modality_to_dataset.items():
            sub = combined[combined["DataSet"] == dataset_name]

            fdr_row = sub[sub["Method"] == "FDR"]
            for mean_col, std_col, _ in _COHORT_PANEL_METRICS:
                mean_val = float(fdr_row[mean_col].iloc[0]) if not fdr_row.empty and mean_col in fdr_row else np.nan
                std_val = (float(fdr_row[std_col].iloc[0])
                           if not fdr_row.empty and std_col in fdr_row and pd.notna(fdr_row[std_col].iloc[0])
                           else 0.0)
                rows.append({
                    "Cohort": cohort, "Modality": modality, "Group": "FDR",
                    "Metric": mean_col, "mean": mean_val, "std": std_val,
                })

            method_sub = sub[sub["Method"].isin(all_methods)]
            for mean_col, std_col, _ in _COHORT_PANEL_METRICS:
                per_method_means = method_sub[["Method", mean_col]].dropna()
                mean_val = float(per_method_means[mean_col].mean()) if not per_method_means.empty else np.nan
                std_val = float(per_method_means[mean_col].std()) if len(per_method_means) > 1 else 0.0
                rows.append({
                    "Cohort": cohort, "Modality": modality, "Group": "All-method mean",
                    "Metric": mean_col, "mean": mean_val, "std": std_val,
                    "N_Methods": len(per_method_means),
                })

    out = pd.DataFrame(rows)
    round_cols = [c for c in ["mean", "std"] if c in out.columns]
    out[round_cols] = out[round_cols].round(decimals)
    return out


def generate_cohort_clinical_vs_multiomics_figure(combined, output_folder, decimals=4):
    """
    ONE figure, three panels (CAU (pp), RRD (pp), Recovery Ratio). Each
    panel's x-axis has four clusters in order: TransNEO FDR, TransNEO
    All-method mean, ARTemis FDR, ARTemis All-method mean. Within each
    cluster the LEFT bar is Clinical and the RIGHT bar is Multi-omics.
    Error bars are the across-run std for FDR and the across-method std for
    the all-method mean. Each bar is labelled with its value, and a
    Delta = Multi-omics - Clinical annotation sits above each cluster.
    """
    data = build_cohort_clinical_vs_multiomics_data(combined, decimals=decimals)

    data_path = os.path.join(output_folder, f"{OUTPUT_FOLDER_NAME}_Clinical_vs_Multiomics_within_cohort.csv")
    data.to_csv(data_path, index=False)
    print(f"[SAVED] {data_path}")

    if data["mean"].isna().all():
        print("  [SKIP] No data available for the within-cohort Clinical vs Multi-omics figure.")
        return

    # x-axis cluster order, exactly as in the reference figure.
    clusters = [(cohort, group)
                for cohort in config.CLINICAL_MULTIOMICS_PAIRS
                for group in ("FDR", "All-method mean")]
    cluster_labels = [f"{cohort}\n{group}" for cohort, group in clusters]
    x = np.arange(len(clusters))
    bar_width = 0.32

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for ax, (mean_col, std_col, metric_label) in zip(axes, _COHORT_PANEL_METRICS):
        metric_data = data[data["Metric"] == mean_col]

        clin_means, clin_stds, multi_means, multi_stds, deltas = [], [], [], [], []
        for cohort, group in clusters:
            g = metric_data[(metric_data["Cohort"] == cohort) & (metric_data["Group"] == group)]
            clin = g[g["Modality"] == "Clinical"]
            multi = g[g["Modality"] == "Multi-omics"]
            c_mean = float(clin["mean"].iloc[0]) if not clin.empty else np.nan
            c_std = float(clin["std"].iloc[0]) if not clin.empty else 0.0
            m_mean = float(multi["mean"].iloc[0]) if not multi.empty else np.nan
            m_std = float(multi["std"].iloc[0]) if not multi.empty else 0.0
            clin_means.append(c_mean)
            clin_stds.append(c_std)
            multi_means.append(m_mean)
            multi_stds.append(m_std)
            deltas.append(m_mean - c_mean if np.isfinite(m_mean) and np.isfinite(c_mean) else np.nan)

        clin_means_arr = np.nan_to_num(np.array(clin_means))
        multi_means_arr = np.nan_to_num(np.array(multi_means))
        clin_stds_arr = np.nan_to_num(np.array(clin_stds))
        multi_stds_arr = np.nan_to_num(np.array(multi_stds))

        bars_clin = ax.bar(x - bar_width / 2, clin_means_arr, bar_width,
                            yerr=clin_stds_arr, capsize=4, label="Clinical",
                            color="lightsteelblue", edgecolor="black")
        bars_multi = ax.bar(x + bar_width / 2, multi_means_arr, bar_width,
                             yerr=multi_stds_arr, capsize=4, label="Multi-omics",
                             color="darkorange", edgecolor="black")

        # Value label on top of each individual bar's error whisker.
        for bar, val, err in zip(bars_clin, clin_means_arr, clin_stds_arr):
            offset = err + 0.02 * max(abs(val) + err, 1.0)
            va = "bottom" if val >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, val + (offset if val >= 0 else -offset),
                    f"{val:.2f}", ha="center", va=va, fontsize=9)
        for bar, val, err in zip(bars_multi, multi_means_arr, multi_stds_arr):
            offset = err + 0.02 * max(abs(val) + err, 1.0)
            va = "bottom" if val >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, val + (offset if val >= 0 else -offset),
                    f"{val:.2f}", ha="center", va=va, fontsize=9)

        # Delta annotation above each (Clinical, Multi-omics) cluster.
        for i, (c_mean, c_std, m_mean, m_std, delta) in enumerate(
            zip(clin_means, clin_stds, multi_means, multi_stds, deltas)
        ):
            if not np.isfinite(delta):
                continue
            top = max(
                (c_mean if np.isfinite(c_mean) else 0.0) + (c_std if np.isfinite(c_std) else 0.0),
                (m_mean if np.isfinite(m_mean) else 0.0) + (m_std if np.isfinite(m_std) else 0.0),
            )
            sign = "+" if delta >= 0 else "\u2212"
            ax.text(x[i], top + 0.10 * max(abs(top), 1.0),
                    f"\u0394={sign}{abs(delta):.2f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color="seagreen")

        if mean_col == "Recovery_Ratio":
            ax.axhline(1.0, linestyle="--", linewidth=1, color="gray", alpha=0.8)
        else:
            ax.axhline(0.0, color="black", linewidth=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(cluster_labels, fontsize=10)
        ax.set_title(metric_label, fontsize=14)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    fig.suptitle("Clinical vs Multi-omics within cohort -- FDR and all-method mean", fontsize=15, y=1.04)
    path = os.path.join(output_folder, f"{OUTPUT_FOLDER_NAME}_Clinical_vs_Multiomics_within_cohort.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] Saved -> {path}")


def build_multiomics_method_ranking(combined, value_col, output_folder, fname_stub, decimals=2):
    """
    Ranks every method by `value_col` averaged across every multi-omics
    dataset (config.MULTIOMICS_DATASETS -- multi_TransNEO, multi_ARTemis,
    multi_Trans_ART, OOD_multi_Trans_ART), with std across those datasets.

    This is intentionally broader than the paired Clinical-vs-Multi-omics
    comparison (build_paired_modality_comparison): it answers "how do
    methods compare on multi-omics data in general", including the
    pooled-cohort and OOD datasets that have no single-cohort clinical-only
    counterpart and so cannot be paired.
    """
    sub = combined[combined["DataSet"].isin(config.MULTIOMICS_DATASETS)]
    ranking = sub.groupby("Method")[value_col].agg(["mean", "std", "count"]).reset_index()
    ranking = ranking.rename(columns={
        "mean": f"{value_col}_mean", "std": f"{value_col}_std", "count": "N_Datasets",
    })
    ranking = ranking.sort_values(f"{value_col}_mean", ascending=False).reset_index(drop=True)
    ranking.insert(0, "Rank", np.arange(1, len(ranking) + 1))

    path = os.path.join(output_folder, f"{OUTPUT_FOLDER_NAME}_{fname_stub}_MultiomicsOnly_MethodRanking.csv")
    ranking.round(decimals).to_csv(path, index=False)
    print(f"[SAVED] {path}")
    return ranking


# ============================================================
# Batch evaluation across all datasets/scenarios
# ============================================================

def evaluate_all_datasets(output_folder):
    """
    Evaluate every dataset in config.DATASET_REGISTRY (both CV and OOD
    scenarios share the same metric definitions and REC-file format, so a
    single evaluation routine covers both), then build combined pivot
    tables, modality comparisons, and method-ranking plots.
    """
    all_ratio_tables = []
    all_statistical_validation_tables = []
    all_smd_summary_tables = []

    for data_name, cfg in config.DATASET_REGISTRY.items():
        print(f"\n{'='*60}")
        print(f"Evaluating dataset: {data_name}  (scenario={cfg['scenario']})")
        print(f"{'='*60}")

        output_path = os.path.join(output_folder, data_name)
        ensure_dir(output_path)

        try:
            result = results_evaluation(data_name, output_path)
        except Exception as e:
            print(f"  [ERROR] {data_name} failed: {e}")
            continue

        if "recovery_summary" in result:
            ratio_df = result["recovery_summary"].copy()
            keep_cols = [
                "Method",
                "Recovery_Ratio", "Recovery_Ratio_std",
                "Recovery_Ratio_ci95_lo", "Recovery_Ratio_ci95_hi",
                "CAU", "CAU_std", "CAU_ci95_lo", "CAU_ci95_hi",
                "CAU_pp", "CAU_pp_std", "CAU_pp_ci95_lo", "CAU_pp_ci95_hi",
                "RRD", "RRD_std", "RRD_ci95_lo", "RRD_ci95_hi",
                "RRD_pp", "RRD_pp_std", "RRD_pp_ci95_lo", "RRD_pp_ci95_hi",
                "extra_recoveries", "extra_recoveries_std",
                "CAU_pp_perm_p",
                "N_Runs", "N_Valid_CAU_Runs",
            ]
            keep_cols = [c for c in keep_cols if c in ratio_df.columns]
            ratio_df = ratio_df[keep_cols].copy()
            ratio_df.insert(0, "DataSet", data_name)
            num_cols = [c for c in keep_cols if c != "Method"]
            ratio_df[num_cols] = ratio_df[num_cols].round(4)
            all_ratio_tables.append(ratio_df)
        else:
            print(f"  [INFO] No recovery summary for '{data_name}' -- excluded from pivot tables.")


        if "statistical_validation" in result:
            all_statistical_validation_tables.append(result["statistical_validation"].copy())
        else:
            print(f"  [INFO] No statistical-validation table for '{data_name}'.")

        if "smd_summary" in result:
            smd_sub = result["smd_summary"].copy()
            smd_sub.insert(0, "DataSet", data_name)
            all_smd_summary_tables.append(smd_sub)
        else:
            print(f"  [INFO] No covariate-balance (SMD) summary for '{data_name}'.")

    if not all_ratio_tables:
        print("\nNo recovery tables generated -- skipping combined output.")
        return

    combined = pd.concat(all_ratio_tables, ignore_index=True)
    combined["Modality"] = combined["DataSet"].map(config.DATASET_MODALITY)
    combined["Dataset"] = combined["DataSet"].map(config.DATASET_NAME_MAP).fillna(combined["DataSet"])

    # ONE long-format table with every metric (mean, std, 95% CI) for every
    # (dataset, method) 
    metrics_path = os.path.join(output_folder, f"{OUTPUT_FOLDER_NAME}_AllDatasets_Metrics.csv")
    combined.to_csv(metrics_path, index=False)
    print(f"\n[SAVED] {metrics_path}  (CAU/RRD/Recovery-Ratio, mean +/- std "
          f"across runs, with 95% bootstrap CI where available)")

    # ---- Cross-dataset summary: mean + 95% bootstrap CI ACROSS ALL
    #      DATASETS, per method, for CAU_pp / RRD_pp / Recovery_Ratio. This
    #      is the table that answers "what is each method's overall CAU/RR/
    #      RRD across the whole benchmark, with a 95% CI on that average" 
    cross_dataset_rows = []
    for value_col in ["CAU_pp", "RRD_pp", "Recovery_Ratio"]:
        if value_col not in combined.columns:
            continue
        per_method = sv.bootstrap_ci_across_datasets(combined, value_col, n_boot=sv.N_BOOTSTRAP)
        if per_method.empty:
            continue
        per_method.insert(1, "Metric", value_col)
        per_method = per_method.rename(columns={
            f"{value_col}_mean": "Mean",
            f"{value_col}_std": "Std_Across_Datasets",
            f"{value_col}_ci95_lo": "CI95_Lo",
            f"{value_col}_ci95_hi": "CI95_Hi",
        })
        cross_dataset_rows.append(per_method)

    if cross_dataset_rows:
        cross_dataset_summary = pd.concat(cross_dataset_rows, ignore_index=True)
        order = [m for m in config.ALL_METHOD_COL_ORDER if m in cross_dataset_summary["Method"].values]
        cross_dataset_summary["Method"] = pd.Categorical(cross_dataset_summary["Method"], categories=order, ordered=True)
        cross_dataset_summary = cross_dataset_summary.sort_values(["Metric", "Method"]).reset_index(drop=True)
        num_cols = [c for c in ["Mean", "Std_Across_Datasets", "CI95_Lo", "CI95_Hi"] if c in cross_dataset_summary.columns]
        cross_dataset_summary[num_cols] = cross_dataset_summary[num_cols].round(4)

        cross_dataset_path = os.path.join(
            output_folder, f"{OUTPUT_FOLDER_NAME}_AllMethods_CrossDataset_Mean_CI95.csv"
        )
        cross_dataset_summary.to_csv(cross_dataset_path, index=False)
        print(f"[SAVED] {cross_dataset_path}  (CAU/RRD/Recovery-Ratio, mean +/- 95% "
              f"bootstrap CI ACROSS ALL {combined['DataSet'].nunique()} DATASETS, per method)")

    # ONE paper-ready wide pivot for the PRIMARY metric only (CAU_pp), as
    # "mean +/- std" strings -- everything else needed is in the long table
    # above; a full pivot-per-metric set is no longer generated.
    if "CAU_pp" in combined.columns and "CAU_pp_std" in combined.columns:
        wide_mean = combined.pivot_table(index="Dataset", columns="Method", values="CAU_pp", aggfunc="first")
        wide_mean = wide_mean.reindex(index=config.DATASET_ROW_ORDER)
        existing = [m for m in config.ALL_METHOD_COL_ORDER if m in wide_mean.columns]
        extra = [m for m in wide_mean.columns if m not in config.ALL_METHOD_COL_ORDER]
        wide_mean = wide_mean.reindex(columns=existing + extra)
        wide_std = combined.pivot_table(index="Dataset", columns="Method", values="CAU_pp_std", aggfunc="first")
        wide_std = wide_std.reindex(index=config.DATASET_ROW_ORDER).reindex(columns=wide_mean.columns)

        fmt_path = os.path.join(output_folder, f"{OUTPUT_FOLDER_NAME}_AllDatasets_CAU_pp_Formatted.csv")
        build_formatted(wide_mean, wide_std, decimals=2).to_csv(fmt_path)
        print(f"[SAVED] {fmt_path}")

    # ---- Clinical vs Multi-omics comparison (primary metric: CAU_pp only) ----
    cau_modality = build_paired_modality_comparison(combined, "CAU_pp", output_folder, "CAU_pp")
    generate_modality_comparison_plot(cau_modality, output_folder, "CAU_pp", metric_label="CAU (pp)")

    # ---- Per-cohort Clinical vs Multi-omics figure: FDR vs. all-method
    #      mean, panels = CAU (pp), RRD (pp), Recovery Ratio (unchanged) ----
    generate_cohort_clinical_vs_multiomics_figure(combined, output_folder)

    # ---- Multi-omics-only method comparison (primary metric: CAU_pp only) ----
    cau_ranking = build_multiomics_method_ranking(combined, "CAU_pp", output_folder, "CAU_pp")
    generate_method_ranking_plot(cau_ranking, "CAU_pp", output_folder, "CAU_pp", metric_label="CAU (pp)")




