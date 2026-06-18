# -*- coding: utf-8 -*-
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

# ============================================================
# Dataset registry
# ============================================================

# Must match STANDARD_MODELS + BASELINE_METHODS in pipeline.py / REC_OOD.py
CV_METHODS = [
    "FDR", "CB", "NN", "RF", "SVR", "XGB", "LR",
    "S_L", "X_L", "DR_L", "R_L",
    "CF", "CUTS", "EP_L", "BITES",
]

OOD_METHODS = [
    "FDR", "CB", "NN", "RF", "SVR", "XGB", "LR",
    "S_L", "X_L", "DR_L", "R_L",
    "CF", "CUTS", "EP_L", "BITES",
]

DATASET_REGISTRY = {
    "GSE41998":                 {"outcome_col": "RCB.category",  "has_pcr": True,  "methods": CV_METHODS},
    "GSE22226":                 {"outcome_col": "RCB_category",  "has_pcr": True,  "methods": CV_METHODS},
    "GSE22358":                 {"outcome_col": "RCB_category",  "has_pcr": False, "methods": CV_METHODS},
    "clin_TransNEO":            {"outcome_col": "RCB.score",     "has_pcr": True,  "methods": CV_METHODS},
    "clin_ARTemis":             {"outcome_col": "RCB.score",     "has_pcr": True,  "methods": CV_METHODS},
    "multi_Trans_ART":          {"outcome_col": "RCB.score",     "has_pcr": True,  "methods": CV_METHODS},
    "multi_TransNEO":           {"outcome_col": "RCB.score",     "has_pcr": True,  "methods": CV_METHODS},
    "multi_ARTemis":            {"outcome_col": "RCB.score",     "has_pcr": True,  "methods": CV_METHODS},
    "OOD_multi_Trans_ART":      {"outcome_col": "RCB.score",     "has_pcr": True,  "methods": OOD_METHODS},
}

DATASET_NAME_MAP = {
    "clin_TransNEO":            "TransNEO clinical",
    "clin_ARTemis":             "ARTemis clinical",
    "GSE41998":                 "GSE41998",
    "GSE22226":                 "GSE22226",
    "clin_GSE22358":            "GSE22358",
    "multi_TransNEO":           "TransNEO multi-omics",
    "multi_ARTemis":            "ARTemis multi-omics",
    "multi_Trans_ART":          "Combined TransNEO + ARTemis multi-omics CV",
    "OOD_multi_Trans_ART":      "TransNEO and ARTemis multi-omics OOD",
}

DATASET_ROW_ORDER = list(DATASET_NAME_MAP.values())

ALL_METHOD_COL_ORDER = [
    "FDR", "CT", "CB", "NN", "LR", "RF", "SVR", "XGB",
    "S_L", "X_L", "DR_L", "R_L",
    "CF", "CUTS", "EP_L", "BITES",
]

MODELS_COLOUR = {
    "FDR":          "blue",
    "CT":           "green",
    "CB":           "cyan",
    "NN":           "yellow",
    "LR":           "orange",
    "RF":           "red",
    "SVR":          "purple",
    "XGB":          "pink",
    "S_L":          "mediumseagreen",
    "X_L":          "steelblue",
    "DR_L":         "tomato",
    "R_L":          "darkorange",
    "CF":           "mediumpurple",
    "CUTS":         "saddlebrown",
    "EP_L":         "teal",
    "BITES":        "deeppink",
}

DATASET_TITLES = {
    "clin_ARTemis":            "ARTemis Clinical Dataset",
    "clin_TransNEO":           "TransNEO Clinical Dataset",
    "GSE41998":                 "GSE41998  Dataset",
    "GSE22226":                 "GSE22226 I-SPY  Dataset",
    "GSE22358":                 "GSE22358  Dataset",
    "multi_ARTemis":           "ARTemis Multi-omics Dataset",
    "multi_TransNEO":          "TransNEO Multi-omics Dataset",
    "multi_Trans_ART":         "Multi-omics Datasets CV: TransNEO and ARTemis",
    "OOD_multi_Trans_ART":     "Multi-omics Datasets OOD: TransNEO train, ARTemis test",
}


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


# ============================================================
# Summary statistics
# ============================================================

def save_summary_stats(summary_dict, method_name, df, outcome_col, has_pcr=True):
    """
    Accumulate per-method recovery and RCB statistics.
    Recovery stats are computed whenever resp.pCR exists in the dataframe,
    regardless of the has_pcr registry flag (column presence is the true gate).
    """
    if "resp.pCR" in df.columns:
        rec_stats = df.groupby("FOLLOW_REC")["resp.pCR"].agg(["count", "sum"]).reset_index()
        for _, row in rec_stats.iterrows():
            summary_dict["recovery"].append({
                "Method":     method_name,
                "FOLLOW_REC": int(row["FOLLOW_REC"]),
                "Count":      int(row["count"]),
                "Recovery":   int(row["sum"]),
            })

    rcb_stats = df.groupby("FOLLOW_REC")[outcome_col].agg(["count", "mean"]).reset_index()
    for _, row in rcb_stats.iterrows():
        summary_dict["rcb"].append({
            "Method":              method_name,
            "FOLLOW_REC":          int(row["FOLLOW_REC"]),
            "Count":               int(row["count"]),
            "Avg_categorical_RCB": round(row["mean"], 4),
        })


# ============================================================
# Core evaluation
# ============================================================

def results_evaluation(data_name, output_path):
    cfg         = DATASET_REGISTRY[data_name]
    outcome_col = cfg["outcome_col"]
    has_pcr     = cfg["has_pcr"]
    methods     = cfg["methods"]

    summary_stats = {"recovery": [], "rcb": []}

    for name in methods:
        rec_file = os.path.join(output_path, f"{data_name}_{name}_REC.csv")
        if not os.path.exists(rec_file):
            print(f"    [SKIP] Missing REC file for '{name}'")
            continue
        df = load_dataset(rec_file)
        save_summary_stats(summary_stats, name, df, outcome_col, has_pcr)

    ct_file = os.path.join(output_path, f"{data_name}_CT_REC.csv")
    if os.path.exists(ct_file):
        save_summary_stats(summary_stats, "CT", load_dataset(ct_file), outcome_col, has_pcr)

    result = {"output_dir": output_path}

    # ---- Recovery summary ----
    if summary_stats["recovery"]:
        raw = pd.DataFrame(summary_stats["recovery"])

        # Guard: pivot requires both FOLLOW_REC=0 and FOLLOW_REC=1 per method.
        # Methods where ALL patients follow or NONE follow will be missing one
        # side after pivot, producing NaN. Fill with 0 so plots don't crash.
        pivot = raw.pivot_table(
            index="Method", columns="FOLLOW_REC",
            values=["Count", "Recovery"], aggfunc="sum", fill_value=0
        )
        pivot.columns = [
            "NotFollowing_Count", "Following_Count",
            "NotFollowing_Recovery", "Following_Recovery",
        ]
        pivot = pivot.reset_index()

        # Compute Recovery_Ratio; guard against division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            follow_rate    = np.where(pivot["Following_Count"]    > 0,
                                      pivot["Following_Recovery"]    / pivot["Following_Count"],    np.nan)
            notfollow_rate = np.where(pivot["NotFollowing_Count"] > 0,
                                      pivot["NotFollowing_Recovery"] / pivot["NotFollowing_Count"], np.nan)
            pivot["Recovery_Ratio"] = np.where(
                notfollow_rate > 0, follow_rate / notfollow_rate, np.nan
            )

        pivot.to_csv(os.path.join(output_path, f"{data_name}_Recovery_Ratio.csv"), index=False)
        generate_recovery_ratio_plot(pivot, output_path, data_name)
        generate_recovery_plot(pivot, output_path, data_name)
        result["recovery_summary"] = pivot
    else:
        print(f"  [INFO] No resp.pCR column found in any REC file for '{data_name}' -- skipping recovery plots.")

    # ---- RCB summary ----
    if summary_stats["rcb"]:
        rcb_raw = pd.DataFrame(summary_stats["rcb"])

        # Use two separate pivots to avoid pandas alphabetical MultiIndex ordering.
        # A single pivot_table(values=["Count","Avg_categorical_RCB"]) sorts columns
        # as ("Avg_categorical_RCB",0/1) then ("Count",0/1) -- the opposite of what
        # the column-rename list assumed, causing counts and averages to be swapped.
        _pv_cnt = rcb_raw.pivot_table(
            index="Method", columns="FOLLOW_REC",
            values="Count", aggfunc="mean", fill_value=0
        ).rename(columns={0: "NotFollowing_Count", 1: "Following_Count"})

        _pv_avg = rcb_raw.pivot_table(
            index="Method", columns="FOLLOW_REC",
            values="Avg_categorical_RCB", aggfunc="mean", fill_value=np.nan
        ).rename(columns={0: "NotFollowing_Avg_RCB", 1: "Following_Avg_RCB"})

        rcb_pivot = pd.concat([_pv_cnt, _pv_avg], axis=1).reset_index()
        rcb_pivot.to_csv(
            os.path.join(output_path, f"{data_name}_RCB_Score_Comparison.csv"), index=False
        )
        generate_rcb_boxplot(rcb_pivot, output_path, data_name)
        result["rcb_summary"] = rcb_pivot

    return result


# ============================================================
# Plotting helpers
# ============================================================

def _figure_size(data_name):
    n = len(DATASET_REGISTRY[data_name]["methods"])
    if n > 7 or "3TS" in data_name:
        return (max(13, n * 1.4), 8)
    return (9, 8)


def _sort_methods(df, col="Method"):
    """Sort df rows by ALL_METHOD_COL_ORDER, keeping only rows with known methods."""
    present = [m for m in ALL_METHOD_COL_ORDER if m in df[col].values]
    df = df.copy()
    df[col] = pd.Categorical(df[col], categories=present, ordered=True)
    return df.sort_values(col).dropna(subset=[col]).reset_index(drop=True)


def generate_recovery_ratio_plot(df, output_path, data_name):
    df = _sort_methods(df)
    # Drop rows with NaN ratio (method had no valid comparison)
    df = df.dropna(subset=["Recovery_Ratio"]).reset_index(drop=True)
    if df.empty:
        print(f"  [SKIP] No valid Recovery_Ratio rows for {data_name}")
        return

    methods = df["Method"].tolist()
    ratios  = df["Recovery_Ratio"].values.astype(float)
    x       = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=_figure_size(data_name))
    bars = ax.bar(x, ratios, width=0.6,
                  color=[MODELS_COLOUR.get(m, "gray") for m in methods],
                  edgecolor="black")

    for bar, ratio in zip(bars, ratios):
        if np.isfinite(ratio):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{ratio:.2f}", ha="center", fontsize=10)

    finite_ratios = ratios[np.isfinite(ratios)]
    ymax = (float(finite_ratios.max()) if len(finite_ratios) > 0 else 1.0) + 0.5

    ax.set_title(f"{DATASET_TITLES.get(data_name, data_name)} -- Recovery Ratio", fontsize=15)
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

    methods          = df["Method"].tolist()
    following        = pd.to_numeric(df["Following_Recovery"],    errors="coerce").fillna(0).values
    not_following    = pd.to_numeric(df["NotFollowing_Recovery"], errors="coerce").fillna(0).values
    following_count  = pd.to_numeric(df["Following_Count"],       errors="coerce").fillna(0).values
    nfollowing_count = pd.to_numeric(df["NotFollowing_Count"],    errors="coerce").fillna(0).values
    x = np.arange(len(methods))

    bar_width = 0.4
    gap       = 0.02
    fig, ax   = plt.subplots(figsize=_figure_size(data_name))

    bars1 = ax.bar(x - (bar_width / 2 + gap / 2), following,     bar_width)
    bars2 = ax.bar(x + (bar_width / 2 + gap / 2), not_following, bar_width)

    for i, method in enumerate(methods):
        colour = MODELS_COLOUR.get(method, "gray")
        for bars, hatch in [(bars1, "+"), (bars2, "/")]:
            bars[i].set_facecolor(colour)
            bars[i].set_edgecolor("black")
            bars[i].set_hatch(hatch)

    for bar, rec, cnt in zip(bars1, following, following_count):
        h = _safe_float(bar.get_height())
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                f"{int(rec)}\ntotal={int(cnt)}", ha="center", fontsize=9)
    for bar, rec, cnt in zip(bars2, not_following, nfollowing_count):
        h = _safe_float(bar.get_height())
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                f"{int(rec)}\ntotal={int(cnt)}", ha="center", fontsize=9)

    legend_patches = [
        Patch(facecolor="gray", hatch="+", label="Following",    edgecolor="black"),
        Patch(facecolor="gray", hatch="/", label="Not Following", edgecolor="black"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=13,
              handlelength=2.5, handleheight=1.8, framealpha=0.6)

    ax.set_title(DATASET_TITLES.get(data_name, data_name), fontsize=15)
    ax.set_ylabel("Number of Recovered Patients", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=13)

    # Guard ymax against NaN/Inf
    all_vals = np.concatenate([following, not_following])
    ymax = _safe_max(all_vals, default=10.0) + 10
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    path = os.path.join(output_path, f"{data_name}_Recovery_comparison.png")
    plt.savefig(path)
    plt.close()
    print(f"  [PLOT] Saved -> {path}")


def generate_rcb_boxplot(df, output_path, data_name):
    df = _sort_methods(df)
    if df.empty:
        return

    data_pts, colours, positions, hatches = [], [], [], []
    base = 1
    xtick_labels = []

    for row in df.itertuples(index=False):
        method = row.Method
        colour = MODELS_COLOUR.get(method, "gray")

        f_cnt  = int(_safe_float(row.Following_Count,    0))
        nf_cnt = int(_safe_float(row.NotFollowing_Count, 0))
        f_avg  = _safe_float(getattr(row, "Following_Avg_RCB",    None), np.nan)
        nf_avg = _safe_float(getattr(row, "NotFollowing_Avg_RCB", None), np.nan)

        # Use a small jitter around the mean; fall back to [0] if no data
        f_scores  = np.random.normal(f_avg,  0.1, max(f_cnt,  1)) if np.isfinite(f_avg)  else np.array([np.nan])
        nf_scores = np.random.normal(nf_avg, 0.1, max(nf_cnt, 1)) if np.isfinite(nf_avg) else np.array([np.nan])

        data_pts.extend([f_scores, nf_scores])
        colours.extend([colour, colour])
        hatches.extend(["+", "/"])
        positions.extend([base, base + 0.6])
        xtick_labels.append(method)
        base += 1.8

    if not data_pts:
        return

    fig, ax = plt.subplots(figsize=_figure_size(data_name))
    bp = ax.boxplot(data_pts, patch_artist=True, widths=0.45, positions=positions,
                    showfliers=False)   # suppress outlier circles -- data is jittered around a mean,
                                        # so fliers are random noise artefacts, not real outliers
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colours[i])
        patch.set_edgecolor("black")   # hatch lines inherit edge colour; must be black to be visible
        patch.set_hatch(hatches[i])
        patch.set_linewidth(1.2)

    xtick_pos = [np.mean(positions[i:i + 2]) for i in range(0, len(positions), 2)]
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=13)

    legend_patches = [
        Patch(facecolor="gray", hatch="+", label="Following",    edgecolor="black"),
        Patch(facecolor="gray", hatch="/", label="Not Following", edgecolor="black"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=13,
              handlelength=2.0, handleheight=2.0, framealpha=0.6)

    ax.set_title(DATASET_TITLES.get(data_name, data_name), fontsize=15)
    ax.set_ylabel("Average RCB Score", fontsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    path = os.path.join(output_path, f"{data_name}_RCB_Score_comparison.png")
    plt.savefig(path)
    plt.close()
    print(f"  [PLOT] Saved -> {path}")


# ============================================================
# CAU metrics
# ============================================================

def _compute_cau(df):
    df = df.copy()

    def safe_div(num, den):
        num = pd.to_numeric(num, errors="coerce")
        den = pd.to_numeric(den, errors="coerce")
        return np.where((den > 0) & np.isfinite(den), num / den, np.nan)

    df["p_rec_follow"] = safe_div(df["Following_Recovery"],    df["Following_Count"])
    df["p_rec_not"]    = safe_div(df["NotFollowing_Recovery"], df["NotFollowing_Count"])
    df["N"]            = df[["Following_Count", "NotFollowing_Count"]].sum(axis=1, min_count=1)
    df["coverage"]     = safe_div(df["Following_Count"], df["N"])
    df["CAU"]          = (df["p_rec_follow"] - df["p_rec_not"]) * df["coverage"]
    df["CAU_pp"]       = df["CAU"] * 100.0
    df["extra_recoveries"] = df["CAU"] * df["N"]
    return df


# ============================================================
# Batch runner
# ============================================================

def run_all_datasets(output_folder):
    all_ratio_tables = []

    for data_name, cfg in DATASET_REGISTRY.items():
        print(f"\n{'='*60}")
        print(f"Evaluating dataset: {data_name}")
        print(f"{'='*60}")

        output_path = os.path.join(output_folder, data_name)
        ensure_dir(output_path)

        try:
            result = results_evaluation(data_name, output_path)
        except Exception as e:
            print(f"  [ERROR] {data_name} failed: {e}")
            continue

        if "recovery_summary" in result:
            ratio_df = _compute_cau(result["recovery_summary"])
            ratio_df = ratio_df[
                ["Method", "Recovery_Ratio", "CAU", "CAU_pp", "extra_recoveries"]
            ].copy()
            ratio_df.insert(0, "DataSet", data_name)
            ratio_df[["Recovery_Ratio", "CAU", "CAU_pp", "extra_recoveries"]] = (
                ratio_df[["Recovery_Ratio", "CAU", "CAU_pp", "extra_recoveries"]].round(4)
            )
            all_ratio_tables.append(ratio_df)
        else:
            print(f"  [INFO] No recovery summary for '{data_name}' -- excluded from pivot tables.")

    if not all_ratio_tables:
        print("\nNo recovery tables generated -- skipping combined output.")
        return

    combined = pd.concat(all_ratio_tables, ignore_index=True)

    rr_path   = os.path.join(output_folder, f"{output_folder_name}_AllDatasets_Recovery_Ratio.csv")
    long_path = os.path.join(output_folder, f"{output_folder_name}_AllDatasets_Metrics_Long.csv")
    combined[["DataSet", "Method", "Recovery_Ratio"]].to_csv(rr_path, index=False)
    combined.to_csv(long_path, index=False)
    print(f"\n[SAVED] {rr_path}")
    print(f"[SAVED] {long_path}")

    combined["Dataset"] = combined["DataSet"].map(DATASET_NAME_MAP).fillna(combined["DataSet"])

    def pivot_and_save(metric_col, fname_stub, decimals=3):
        wide = combined.pivot_table(
            index="Dataset", columns="Method",
            values=metric_col, aggfunc="first"
        )
        wide = wide.reindex(index=DATASET_ROW_ORDER)
        existing = [m for m in ALL_METHOD_COL_ORDER if m in wide.columns]
        extra    = [m for m in wide.columns if m not in ALL_METHOD_COL_ORDER]
        wide     = wide.reindex(columns=existing + extra).round(decimals)
        path     = os.path.join(output_folder, f"{output_folder_name}_AllDatasets_{fname_stub}_Pivot.csv")
        wide.to_csv(path)
        print(f"[SAVED] {path}")

    pivot_and_save("Recovery_Ratio",   "Recovery_Ratio")
    pivot_and_save("CAU",              "CAU",             decimals=4)
    pivot_and_save("CAU_pp",           "CAU_pp",          decimals=2)
    pivot_and_save("extra_recoveries", "ExtraRecoveries", decimals=2)


# ============================================================
# Entry point
# ============================================================

base_path          = os.getcwd()
output_folder_name = "output"
output_folder      = os.path.join(base_path, output_folder_name)
ensure_dir(output_folder)

if __name__ == "__main__":
    run_all_datasets(output_folder)