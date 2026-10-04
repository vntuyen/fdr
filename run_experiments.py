# -*- coding: utf-8 -*-
"""
run_experiments.py
===================
Top-level entry point: runs FDR + all baseline methods on every enabled
dataset in config.DATASET_REGISTRY, each under its own evaluation protocol
("cv", "ood" or "splits"), then runs the full evaluation suite (CAU, RRD,
Recovery Ratio, outcome summaries, statistical validation).

Every dataset is declared in config.DATASET_REGISTRY; adding one means
putting its CSV(s) in input/<name>/ and adding one entry there.

Usage
-----
    python run_experiments.py                  # run everything, then evaluate
    python run_experiments.py --list-datasets  # show the registered datasets
    python run_experiments.py --no-run         # evaluation only (reuse existing output/)
    python run_experiments.py --no-eval        # run methods only, skip evaluation
    python run_experiments.py --datasets multi_ARTemis
    python run_experiments.py --methods FDR CB CTR
"""

import os
import argparse
import numpy as np
import pandas as pd

import config
import fdr
import baselines
import evaluation


def get_all_methods() -> dict:
    """{"FDR": <placeholder, resolved per-scenario>} + every baseline."""
    methods = {"FDR": None}
    methods.update(baselines.BASELINE_METHODS)
    return methods


def _drop_torch_methods_if_unavailable(methods: dict) -> dict:
    """
    If torch could not be imported (config.TORCH_AVAILABLE is False), remove
    the methods that need it (config.TORCH_METHODS: FDR, BITES) instead of
    letting them re-import a broken torch mid-run, which can abort the whole
    Python process on Windows ("Unhandled exception caught in
    c10/util/AbortHandler.h"). Every other method still runs.
    """
    if config.TORCH_AVAILABLE:
        return methods
    skipped = [m for m in methods if m in config.TORCH_METHODS]
    if skipped:
        print(
            f"  [WARN] torch is not importable ({config._TORCH_IMPORT_ERROR!r}).\n"
            f"         Skipping {skipped}; reinstall torch to run them. "
            "Other methods continue."
        )
    return {m: fn for m, fn in methods.items() if m not in config.TORCH_METHODS}


def _methods_for_scenario(methods: dict, scenario: str) -> dict:
    """Bind the FDR placeholder to the scenario-appropriate implementation
    ("splits" uses FDR's CV variant: one fit per training set)."""
    resolved = _drop_torch_methods_if_unavailable(dict(methods))
    if "FDR" in resolved:
        resolved["FDR"] = fdr.get_fdr_fn(scenario)
    return resolved


def compare_recommendations(recommended_df, original_df, outcome_col, tp_cols, propensity=None):
    """
    Builds the per-patient REC dataframe: original covariates/outcome +
    per-arm Q-value predictions (already in recommended_df) + REC_TP +
    CURRENT_TP (actually-received arm) + FOLLOW_REC (legacy diagnostic flag).


    """
    recommended_df = recommended_df.copy()
    recommended_df["CURRENT_TP"] = original_df[tp_cols].idxmax(axis=1).values
    recommended_df["FOLLOW_REC"] = recommended_df["REC_TP"] == recommended_df["CURRENT_TP"]

    if propensity is not None:
        for i, tp in enumerate(tp_cols):
            recommended_df[f"PROP_{tp}"] = propensity[:, i]

    combined = pd.concat(
        [original_df.reset_index(drop=True), recommended_df.reset_index(drop=True)],
        axis=1,
    )
    return combined, None, None



from sklearn.linear_model import LogisticRegression
PROPENSITY_CLIP = 1e-3

class PropensityModel:
    """
    Multinomial logistic regression propensity model: P(T = a | X) for each
    treatment arm a, fit on TRAINING data only.

    Usage
    -----
        prop = PropensityModel(treatment_plans).fit(X_train)
        e_hat = prop.predict_proba(X_test)   # (n_test, n_arms) array,
                                              # columns ordered as treatment_plans
    """

    def __init__(self, treatment_plans: list, max_iter: int = 1000, seed: int = 42):
        self.treatment_plans = list(treatment_plans)
        self.max_iter = max_iter
        self.seed = seed
        self._model = None
        self._feature_cols = None

    def fit(self, X: pd.DataFrame) -> "PropensityModel":
        """
        X must contain the one-hot treatment-plan indicator columns; the
        arm actually received by each row is recovered as the argmax over
        those columns. Covariates are every other column.
        """
        self._feature_cols = [c for c in X.columns if c not in self.treatment_plans]
        tp_cols = [tp for tp in self.treatment_plans if tp in X.columns]
        if len(tp_cols) < 2:
            raise ValueError(
                f"PropensityModel needs >= 2 treatment-plan columns present in X; "
                f"got {tp_cols}."
            )
        t_idx = X[tp_cols].values.argmax(axis=1)
        # Map local tp_cols ordering back to the full treatment_plans ordering.
        local_to_global = [self.treatment_plans.index(tp) for tp in tp_cols]
        t_global = np.array([local_to_global[i] for i in t_idx])

        self._model = LogisticRegression(
            max_iter=self.max_iter, random_state=self.seed,
        )
        self._model.fit(X[self._feature_cols].fillna(0).values, t_global)
        # Classes seen during fit (may be a subset of treatment_plans if an
        # arm is entirely absent from the training fold).
        self._classes_ = list(self._model.classes_)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns an (n, len(treatment_plans)) array of P(T=a|X) for every
        arm in self.treatment_plans, filling unseen-in-training arms with a
        small floor probability rather than crashing."""
        if self._model is None:
            raise RuntimeError("Call fit() before predict_proba().")
        raw = self._model.predict_proba(X[self._feature_cols].fillna(0).values)
        out = np.full((len(X), len(self.treatment_plans)), PROPENSITY_CLIP)
        for j, cls in enumerate(self._classes_):
            out[:, cls] = raw[:, j]
        # Renormalise rows to sum to 1 after inserting the floor probabilities.
        out = out / out.sum(axis=1, keepdims=True)
        return out


def fit_propensity_model(X_train: pd.DataFrame, treatment_plans: list, seed: int = 42) -> PropensityModel:
    """Convenience wrapper: fit a PropensityModel on a training fold/split."""
    return PropensityModel(treatment_plans, seed=seed).fit(X_train)



# ============================================================
# Shared helpers
# ============================================================

def prepare_xy(df: pd.DataFrame, data_name: str):
    """Features X and raw outcome y for one dataset file. The recovery
    column (e.g. resp.pCR) is always excluded from the features, even when
    the spec forgets to list it in remove_cols."""
    cfg = config.DATASET_REGISTRY[data_name]
    drop = list(dict.fromkeys(cfg["remove_cols"] + [cfg["recovery_col"]]))
    return config.preprocess_data(df, cfg["outcome_col"], drop, cfg["treatment_plans"])


def check_outcome_leakage(X: pd.DataFrame, y: pd.Series, data_name: str,
                          treatment_plans: list, threshold: float = 0.9):
    """Warn about features that are almost a copy of the outcome (|r| >=
    threshold), e.g. RCB.category next to RCB.score. Warns only; add such
    columns to remove_cols in the dataset spec."""
    feats = [c for c in X.columns if c not in treatment_plans]
    if not feats or y.nunique() < 2:
        return []
    r = X[feats].apply(lambda col: np.corrcoef(col, y)[0, 1] if col.std() > 0 else 0.0)
    leaky = r[r.abs() >= threshold].round(3)
    if len(leaky):
        print(f"  [WARN] {data_name}: features nearly identical to the outcome "
              f"(|r| >= {threshold}): {leaky.to_dict()}. Add them to remove_cols.")
    return list(leaky.index)


def _align_columns(X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
    """Give a separately preprocessed test set exactly the training columns."""
    missing = [c for c in X_train.columns if c not in X_test.columns]
    if missing:
        print(f"    [WARN] {len(missing)} training columns missing in test set; filled with 0: {missing[:5]}")
    return X_test.reindex(columns=X_train.columns, fill_value=0)


def _run_methods_on_split(methods, data_name, X_train, y_train, X_test, test_df,
                          propensity_test, seed):
    """Run every method on one train/test split; return {method: REC df}.

    Methods always MINIMISE: they are trained on outcome_sign * y (so a
    higher-is-better outcome such as pCR is negated), and their per-arm
    columns are converted back to the outcome's own scale before saving.
    """
    cfg = config.DATASET_REGISTRY[data_name]
    tps = cfg["treatment_plans"]
    sign = config.outcome_sign(data_name)
    y_fit = sign * y_train
    out = {}
    for name, recommend_fn in methods.items():
        try:
            X_tr_m, X_te_m = config.preprocess_df_for_model(
                model_name=name, data_name=data_name, X_train=X_train, X_test=X_test,
                treatment_plans=tps, verbose=False,
            )
            y_tr_m = y_fit
            if config.should_balance(name):
                ratio = config.get_balance_config(data_name)
                if ratio is not None:
                    X_tr_m, y_tr_m = config.balance_training_arms(
                        X_train=X_tr_m, y_train=y_fit, treatment_plans=tps,
                        max_ratio=ratio, seed=seed)
            rec_df = recommend_fn(X_tr_m, y_tr_m, X_te_m, tps)
            rec_df[tps] = rec_df[tps] * sign   # back to the outcome's own scale
            combined_df, _, _ = compare_recommendations(
                rec_df, test_df, cfg["outcome_col"], tps, propensity=propensity_test)
            out[name] = combined_df
        except Exception as e:
            print(f"    [WARN] Method '{name}' failed: {type(e).__name__}: {e}")
    return out


def _save_all_runs(per_run_results, output_path, data_name, n_runs):
    for name, df_list in per_run_results.items():
        if not df_list:
            print(f"  [SKIP] No results for method '{name}'")
            continue
        all_runs_df = pd.concat(df_list, ignore_index=True)
        out_file = os.path.join(output_path, f"{data_name}_{name}_REC_all_runs.csv")
        all_runs_df.to_csv(out_file, index=False)
        print(f"  [SAVED] {out_file}  ({n_runs} runs combined)")


# ============================================================
# Scenario "cv": repeated stratified k-fold cross-validation
# ============================================================

def make_recommendations_cv(methods, data_name, input_path, output_path, seed=42, k=5):
    cfg = config.DATASET_REGISTRY[data_name]
    tps = cfg["treatment_plans"]
    df = config.load_dataset(os.path.join(input_path, cfg["files"]["data"]))
    X_full, y_full = prepare_xy(df, data_name)
    df = df.loc[X_full.index]

    split_iter, arm_labels, tp_cols_present = config.make_cv_splitter(X_full, df, tps, k, seed)
    all_rec_dfs = {name: [] for name in methods}

    for fold, (train_idx, test_idx) in enumerate(split_iter):
        train_dist = {tp: int((arm_labels[train_idx] == i).sum()) for i, tp in enumerate(tp_cols_present)}
        test_dist = {tp: int((arm_labels[test_idx] == i).sum()) for i, tp in enumerate(tp_cols_present)}
        print(f"  Fold {fold + 1}/{k}  |  train={train_dist}  test={test_dist}")

        X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
        y_train = y_full.iloc[train_idx]
        try:
            propensity_test = fit_propensity_model(X_train, tps, seed=seed).predict_proba(X_test)
        except Exception as e:
            print(f"    [WARN] Propensity model failed on fold {fold + 1}: {e}")
            propensity_test = None

        fold_out = _run_methods_on_split(methods, data_name, X_train, y_train, X_test,
                                         df.iloc[test_idx], propensity_test, seed)
        for name, rec in fold_out.items():
            all_rec_dfs[name].append(rec)

    final_dfs = {}
    for name, df_list in all_rec_dfs.items():
        if not df_list:
            continue
        n_folds = len(df_list)
        if n_folds < k:
            print(f"  [WARN] '{name}' produced results for only {n_folds}/{k} folds.")
        final_df = pd.concat(df_list, ignore_index=True)
        final_df.to_csv(os.path.join(output_path, f"{data_name}_{name}_REC.csv"), index=False)
        final_dfs[name] = final_df
    return final_dfs


def run_cv_pipeline_entry(data_name, base_path=None, methods=None, k=None,
                          n_repeats=None, repeat_seeds=None, **_):
    """
    Runs the k-fold CV protocol `n_repeats` times (config.REPEAT_SEEDS), each
    with a different seed for both the split and every model's randomness,
    and saves one <dataset>_<method>_REC_all_runs.csv per method.
    """
    base_path = base_path or os.getcwd()
    cfg = config.DATASET_REGISTRY[data_name]
    k = k or cfg.get("cv_k", 5)
    repeat_seeds = repeat_seeds or config.REPEAT_SEEDS[:n_repeats or len(config.REPEAT_SEEDS)]
    methods = _methods_for_scenario(methods or get_all_methods(), "cv")

    input_path = config.dataset_input_dir(data_name, base_path)
    output_path = os.path.join(base_path, "output", data_name)
    config.ensure_dir(output_path)

    _df = config.load_dataset(os.path.join(input_path, cfg["files"]["data"]))
    _X, _y = prepare_xy(_df, data_name)
    check_outcome_leakage(_X, _y, data_name, cfg["treatment_plans"])

    per_run_results = {name: [] for name in methods}
    for run_idx, run_seed in enumerate(repeat_seeds, start=1):
        config.set_seed(run_seed)
        run_output_path = os.path.join(output_path, f"run{run_idx}")
        config.ensure_dir(run_output_path)
        print(f"\n{'='*60}\nDataset: {data_name}  |  Run {run_idx}/{len(repeat_seeds)}  |  seed={run_seed}")
        print(f"Methods: {list(methods.keys())}")
        print(f"Split: {k}-fold StratifiedKFold (by arm)  |  each patient tested exactly once\n{'='*60}")
        final_dfs = make_recommendations_cv(methods, data_name, input_path, run_output_path,
                                            seed=run_seed, k=k)
        for name, rec_df in final_dfs.items():
            rec_df = rec_df.copy()
            rec_df["Run"] = run_idx
            rec_df["Seed"] = run_seed
            per_run_results[name].append(rec_df)

    _save_all_runs(per_run_results, output_path, data_name, len(repeat_seeds))
    return per_run_results


# ============================================================
# Scenario "ood": one fixed train/test split (different cohorts)
# ============================================================

def run_ood_pipeline_entry(data_name, base_path=None, methods=None, n_repeats=None,
                           repeat_seeds=None, **_):
    """
    Out-of-distribution protocol: a fixed train/test split where the test
    cohort is a different study. The split is the same in every repeat;
    each repeat (one per seed in config.REPEAT_SEEDS) re-trains every method
    with that seed, so the OOD results carry run-to-run variation (mean +/-
    SD) like the CV results. The propensity model is fit on the TEST cohort,
    because OOD evaluation needs the treatment-assignment mechanism of the
    data being evaluated.
    """
    base_path = base_path or os.getcwd()
    cfg = config.DATASET_REGISTRY[data_name]
    tps = cfg["treatment_plans"]
    repeat_seeds = repeat_seeds or config.REPEAT_SEEDS[:n_repeats or len(config.REPEAT_SEEDS)]
    methods = _methods_for_scenario(methods or get_all_methods(), "ood")

    input_path = config.dataset_input_dir(data_name, base_path)
    output_path = os.path.join(base_path, "output", data_name)
    config.ensure_dir(output_path)

    train_df = config.load_dataset(os.path.join(input_path, cfg["files"]["train"]))
    test_df = config.load_dataset(os.path.join(input_path, cfg["files"]["test"]))
    X_train, y_train = prepare_xy(train_df, data_name)
    X_test, _ = prepare_xy(test_df, data_name)
    X_test = _align_columns(X_train, X_test)
    test_df = test_df.loc[X_test.index]
    check_outcome_leakage(X_train, y_train, data_name, tps)
    print(f"  Train: {X_train.shape}  Test: {X_test.shape}")

    try:
        propensity_test = fit_propensity_model(X_test, tps, seed=config.SEED).predict_proba(X_test)
    except Exception as e:
        print(f"  [WARN] Propensity model failed for OOD test cohort: {e}")
        propensity_test = None

    per_run_results = {name: [] for name in methods}
    for run_idx, run_seed in enumerate(repeat_seeds, start=1):
        config.set_seed(run_seed)
        run_output_path = os.path.join(output_path, f"run{run_idx}")
        config.ensure_dir(run_output_path)
        print(f"\n{'='*60}\nDataset: {data_name} (OOD)  |  Run {run_idx}/{len(repeat_seeds)}  |  seed={run_seed}")
        print(f"Methods: {list(methods.keys())}\n{'='*60}")
        out = _run_methods_on_split(methods, data_name, X_train, y_train, X_test,
                                    test_df, propensity_test, run_seed)
        for name, rec in out.items():
            rec.to_csv(os.path.join(run_output_path, f"{data_name}_{name}_REC.csv"), index=False)
            rec = rec.copy()
            rec["Run"] = run_idx
            rec["Seed"] = run_seed
            per_run_results[name].append(rec)

    config.set_seed(config.SEED)
    _save_all_runs(per_run_results, output_path, data_name, len(repeat_seeds))
    return per_run_results


# ============================================================
# Scenario "splits": several fixed train/test file pairs
# ============================================================

def run_splits_pipeline_entry(data_name, base_path=None, methods=None, **_):
    """
    Runs every method on each predefined train/test pair (files.train /
    files.test with "{k}" = 1..files.n_splits), e.g. the 5 splits shipped
    with the CTR repository. Each split is stored as one "Run", so the
    evaluation treats splits like repeated runs. Propensity models are fit
    on each split's training set, as in CV.
    """
    base_path = base_path or os.getcwd()
    cfg = config.DATASET_REGISTRY[data_name]
    tps = cfg["treatment_plans"]
    files = cfg["files"]
    methods = _methods_for_scenario(methods or get_all_methods(), "splits")

    input_path = config.dataset_input_dir(data_name, base_path)
    output_path = os.path.join(base_path, "output", data_name)
    config.ensure_dir(output_path)

    per_run_results = {name: [] for name in methods}
    n_splits = int(files["n_splits"])
    for k in range(1, n_splits + 1):
        config.set_seed(config.SEED)
        train_df = config.load_dataset(os.path.join(input_path, files["train"].format(k=k)))
        test_df = config.load_dataset(os.path.join(input_path, files["test"].format(k=k)))
        X_train, y_train = prepare_xy(train_df, data_name)
        X_test, _ = prepare_xy(test_df, data_name)
        X_test = _align_columns(X_train, X_test)
        test_df = test_df.loc[X_test.index]
        if k == 1:
            check_outcome_leakage(X_train, y_train, data_name, tps)
        print(f"\n  Split {k}/{n_splits}  |  train={X_train.shape}  test={X_test.shape}")
        try:
            propensity_test = fit_propensity_model(X_train, tps, seed=config.SEED).predict_proba(X_test)
        except Exception as e:
            print(f"    [WARN] Propensity model failed on split {k}: {e}")
            propensity_test = None
        out = _run_methods_on_split(methods, data_name, X_train, y_train, X_test,
                                    test_df, propensity_test, config.SEED)
        for name, rec in out.items():
            rec = rec.copy()
            rec["Run"] = k
            rec["Seed"] = config.SEED
            per_run_results[name].append(rec)

    _save_all_runs(per_run_results, output_path, data_name, n_splits)
    return per_run_results


# One runner per scenario. A new protocol = a new function + an entry here
# (and its name in config.SCENARIOS).
SCENARIO_RUNNERS = {
    "cv": run_cv_pipeline_entry,
    "ood": run_ood_pipeline_entry,
    "splits": run_splits_pipeline_entry,
}


# ============================================================
# Top-level driver
# ============================================================

def resolve_datasets(datasets=None, groups=None) -> list:
    """Dataset names to run, from explicit names and/or groups (default: all)."""
    names = list(config.DATASET_REGISTRY)
    if not datasets and not groups:
        return names
    chosen = []
    for d in datasets or []:
        if d not in config.DATASET_REGISTRY:
            if d in config.ALL_DATASET_DEFINITIONS:
                raise SystemExit(f"Dataset '{d}' is disabled: set \"enabled\": True for it "
                                 "in config.DATASET_REGISTRY.")
            raise SystemExit(f"Unknown dataset '{d}'. Enabled: {names}")
        chosen.append(d)
    for g in groups or []:
        members = [d for d in names if config.DATASET_GROUPS[d] == g]
        if not members:
            raise SystemExit(f"Unknown group '{g}'. Groups: {sorted(set(config.DATASET_GROUPS.values()))}")
        chosen.extend(members)
    return list(dict.fromkeys(chosen))


def run_all_experiments(datasets=None, methods=None, k=None, n_repeats=None,
                        repeat_seeds=None, base_path=None):
    """Runs FDR + all baselines for every selected dataset under its own protocol."""
    base_path = base_path or os.getcwd()
    for data_name in datasets or list(config.DATASET_REGISTRY):
        scenario = config.DATASET_REGISTRY[data_name]["scenario"]
        print(f"\nRunning experiments for dataset: {data_name}  (scenario={scenario})")
        SCENARIO_RUNNERS[scenario](
            data_name=data_name, base_path=base_path, methods=methods,
            k=k, n_repeats=n_repeats, repeat_seeds=repeat_seeds,
        )


def print_dataset_table():
    print(f"Repeats: {len(config.REPEAT_SEEDS)} (seeds {config.REPEAT_SEEDS}) for CV and OOD\n")
    print(f"{'name':24} {'enabled':8} {'group':18} {'scenario':8} {'outcome':12} {'better':7} {'arms':4}  input")
    for d, c in config.ALL_DATASET_DEFINITIONS.items():
        better = "lower" if c["lower_outcome_is_better"] else "higher"
        print(f"{d:24} {str(c.get('enabled', True)):8} {c.get('group', d):18} {c['scenario']:8} "
              f"{c['outcome_col']:12} {better:7} {len(c['treatment_plans']):4}  "
              f"input/{c.get('input_dir', d)}/")


# ============================================================
# CLI entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Run FDR + baselines on every registered dataset, then evaluate.")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Dataset names to run (default: every enabled dataset in config.py).")
    parser.add_argument("--groups", nargs="*", default=None,
                        help="Run every dataset in these groups (e.g. TransNEO_ARTemis).")
    parser.add_argument("--methods", nargs="*", default=None,
                        help="Subset of method names to run (default: FDR + all baselines).")
    parser.add_argument("--k", type=int, default=None, help="CV folds (default: the spec's cv_k, else 5).")
    parser.add_argument("--n-repeats", type=int, default=len(config.REPEAT_SEEDS),
                        help="Repeated CV and OOD runs, one per seed in config.REPEAT_SEEDS (default: all).")
    parser.add_argument("--no-run", action="store_true", help="Skip running methods; evaluate existing output/ only.")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation; only run methods.")
    parser.add_argument("--list-datasets", action="store_true", help="Print the registered datasets and exit.")
    parser.add_argument("--base-path", default=None, help="Base path containing input/ and output/ (default: cwd).")
    args = parser.parse_args()

    if args.list_datasets:
        print_dataset_table()
        return

    base_path = args.base_path or os.getcwd()
    datasets = resolve_datasets(args.datasets, args.groups)

    methods = None
    if args.methods:
        all_methods = get_all_methods()
        methods = {m: all_methods[m] for m in args.methods if m in all_methods}
        missing = [m for m in args.methods if m not in all_methods]
        if missing:
            print(f"  [WARN] Unknown methods ignored: {missing}")

    if not args.no_run:
        run_all_experiments(datasets=datasets, methods=methods, k=args.k,
                            n_repeats=args.n_repeats, base_path=base_path)

    if not args.no_eval:
        output_folder = os.path.join(base_path, evaluation.OUTPUT_FOLDER_NAME)
        config.ensure_dir(output_folder)
        evaluation.evaluate_all_datasets(output_folder)


if __name__ == "__main__":
    main()
