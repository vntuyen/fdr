# -*- coding: utf-8 -*-
"""
run_experiments.py
===================
Top-level entry point: runs FDR + all baseline methods across BOTH
scenarios (CV and OOD) for every configured dataset, then runs the full
evaluation suite (CAU, RR, RRD, average RCB per dataset) over all results.

Usage
-----
    python run_experiments.py                 # run everything, then evaluate
    python run_experiments.py --no-run         # evaluation only (reuse existing output/)
    python run_experiments.py --no-eval        # run methods only, skip evaluation
    python run_experiments.py --datasets multi_ARTemis OOD_multi_Trans_ART
    python run_experiments.py --methods FDR LR RF
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


def _methods_for_scenario(methods: dict, scenario: str) -> dict:
    """Bind the FDR placeholder to the scenario-appropriate implementation."""
    resolved = dict(methods)
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
# CV scenario: repeated k-fold cross-validation
# ============================================================

def make_recommendations_cv(
    methods: dict,
    data_name: str,
    treatment_plans: list,
    outcome_col: str,
    remove_cols: list,
    input_path: str,
    output_path: str,
    seed: int = 42,
    k: int = 5,
):
    input_file = os.path.join(input_path, f"{data_name}.csv")
    df = config.load_dataset(input_file)
    X_full, y_full = config.preprocess_data(df, outcome_col, remove_cols, treatment_plans)

    split_iter, arm_labels, tp_cols_present = config.make_cv_splitter(
        X_full, df, treatment_plans, k, seed
    )

    all_rec_dfs = {name: [] for name in methods}

    for fold, (train_idx, test_idx) in enumerate(split_iter):
        train_dist = {tp: int((arm_labels[train_idx] == i).sum())
                      for i, tp in enumerate(tp_cols_present)}
        test_dist = {tp: int((arm_labels[test_idx] == i).sum())
                     for i, tp in enumerate(tp_cols_present)}
        print(f"  Fold {fold + 1}/{k}  |  train={train_dist}  test={test_dist}")

        X_train = X_full.iloc[train_idx]
        X_test = X_full.iloc[test_idx]
        y_train = y_full.iloc[train_idx]

        # Propensity model: P(actual arm | covariates), fit ONCE per fold on
        # TRAINING data only, then scored on the held-out test fold. Shared
        # across every method evaluated on this fold -- the propensity of
        # the *actually received* arm given covariates does not depend on
        # which recommendation method we're scoring, so refitting it per
        # method would be wasteful and would not change the result.
        try:
            prop_model = fit_propensity_model(X_train, treatment_plans, seed=seed)
            propensity_test = prop_model.predict_proba(X_test)
        except Exception as e:
            print(f"    [WARN] Propensity model failed on fold {fold + 1}: {e}")
            propensity_test = None

        for name, recommend_fn in methods.items():
            try:
                # Step 1: column selection (FDR / LR / SVR / NN only).
                X_tr_m, X_te_m = config.preprocess_df_for_model(
                    model_name=name,
                    data_name=data_name,
                    X_train=X_train,
                    X_test=X_test,
                    treatment_plans=treatment_plans,
                    verbose=False,
                )

                # Step 2: arm balancing (LR / SVR / NN only -- never FDR).
                y_tr_m = y_train
                if config.should_balance(name):
                    balance_ratio = config.get_balance_config(data_name)
                    if balance_ratio is not None:
                        X_tr_m, y_tr_m = config.balance_training_arms(
                            X_train=X_tr_m,
                            y_train=y_train,
                            treatment_plans=treatment_plans,
                            max_ratio=balance_ratio,
                            seed=seed,
                        )

                rec_df = recommend_fn(X_tr_m, y_tr_m, X_te_m, treatment_plans)
                combined_df, _, _ = compare_recommendations(
                    rec_df, df.iloc[test_idx], outcome_col, treatment_plans,
                    propensity=propensity_test,
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


def run_cv_pipeline_entry(
    data_name,
    treatment_plans,
    outcome_col,
    remove_cols,
    base_path: str = None,
    methods: dict = None,
    k: int = 5,
    n_repeats: int = 10,
    repeat_seeds: list = None,
):
    """
    Runs the k-fold CV protocol `n_repeats` times (10 stratified resamples
    by default -- see config.REPEAT_SEEDS), each with a different seed, so
    the reported CAU/RR/RRD have run-to-run variation to report (mean +/-
    std, and a 95% bootstrap CI across all datasets -- see
    statistical_validation.bootstrap_ci_across_datasets) rather than a
    single point estimate.

    Each repeat varies BOTH the k-fold split AND every model's internal
    randomness, since all model factories read the module-level
    `config.SEED` at call time, updated here before each repeat.
    """
    base_path = base_path or os.getcwd()

    if repeat_seeds is None:
        repeat_seeds = config.REPEAT_SEEDS[:n_repeats]
    n_repeats = len(repeat_seeds)

    if methods is None:
        methods = get_all_methods()
    methods = _methods_for_scenario(methods, "cv")

    base_input_path = os.path.join(base_path, "input", data_name)
    base_output_path = os.path.join(base_path, "output", data_name)
    config.ensure_dir(base_output_path)

    per_run_results = {name: [] for name in methods}

    for run_idx, run_seed in enumerate(repeat_seeds, start=1):
        config.set_seed(run_seed)  # propagates to every model factory at call time

        run_output_path = os.path.join(base_output_path, f"run{run_idx}")
        config.ensure_dir(run_output_path)

        print(f"\n{'='*60}")
        print(f"Dataset: {data_name}  |  Run {run_idx}/{n_repeats}  |  seed={run_seed}")
        print(f"Methods: {list(methods.keys())}")
        print(f"Split: {k}-fold StratifiedKFold (by arm)  |  each patient tested exactly once")
        print(f"{'='*60}")

        final_dfs = make_recommendations_cv(
            methods, data_name, treatment_plans, outcome_col,
            remove_cols, base_input_path, run_output_path, seed=run_seed, k=k,
        )
        for name, rec_df in final_dfs.items():
            rec_df = rec_df.copy()
            rec_df["Run"] = run_idx
            rec_df["Seed"] = run_seed
            per_run_results[name].append(rec_df)

    for name, df_list in per_run_results.items():
        if not df_list:
            continue
        all_runs_df = pd.concat(df_list, ignore_index=True)
        out_file = os.path.join(base_output_path, f"{data_name}_{name}_REC_all_runs.csv")
        all_runs_df.to_csv(out_file, index=False)
        print(f"  [SAVED] {out_file}  ({n_repeats} runs combined)")

    return per_run_results


# ============================================================
# OOD scenario: fixed train/test split (different cohorts)
# ============================================================

def run_ood_pipeline_entry(
    data_name,
    treatment_plans,
    outcome_col,
    remove_cols,
    base_path: str = None,
    methods: dict = None,
):
    """
    Runs every method ONCE on a fixed train/test split where the test
    cohort is a different study to the training cohort (out-of-distribution
    generalisation). No outer repeated-run loop -- FDR's own OOD variant
    supplies its run-to-run robustness via an internal seed ensemble
    (see fdr.recommend_fdr_ood).
    """
    base_path = base_path or os.getcwd()

    if methods is None:
        methods = get_all_methods()
    methods = _methods_for_scenario(methods, "ood")

    input_path = os.path.join(base_path, "input", data_name)
    output_path = os.path.join(base_path, "output", data_name)
    config.ensure_dir(output_path)

    train_file = os.path.join(input_path, f"{data_name}_train.csv")
    test_file = os.path.join(input_path, f"{data_name}_test.csv")

    print("  Loading data ...")
    X_train, y_train = config.preprocess_data(
        config.load_dataset(train_file), outcome_col, remove_cols, treatment_plans
    )
    test_data = config.load_dataset(test_file)
    X_test, y_test = config.preprocess_data(test_data, outcome_col, remove_cols, treatment_plans)
    print(f"  Train: {X_train.shape}  Test: {X_test.shape}")

    summary_stats = {"recovery": [], "rcb": []}

    # Propensity model: P(actual arm | covariates). For the OOD scenario the
    # quantity we need to de-confound is "how was treatment actually assigned
    # in the TEST cohort" (that's where FOLLOW_REC / IPW / DR are evaluated),
    # so the propensity model is fit on the TEST cohort's own covariates and
    # actual-arm labels, not on the training cohort -- a propensity model
    # fit on the training cohort would describe assignment mechanics of a
    # different study population and would not correctly reweight the test
    # cohort's confounding. This is still legitimate off-policy evaluation:
    # OPE only requires e(x) to model assignment in the data being evaluated,
    # it does not need to be "held out" from that data the way an outcome
    # model would (we are not using it to predict Y).
    try:
        prop_model_ood = fit_propensity_model(X_test, treatment_plans, seed=config.SEED)
        propensity_test_ood = prop_model_ood.predict_proba(X_test)
    except Exception as e:
        print(f"  [WARN] Propensity model failed for OOD test cohort: {e}")
        propensity_test_ood = None

    for name, recommend_fn in methods.items():
        print(f"    [{name}] running ...")
        try:
            # Step 1: column selection (FDR / LR / SVR / NN only).
            X_tr_m, X_te_m = config.preprocess_df_for_model(
                model_name=name,
                data_name=data_name,
                X_train=X_train,
                X_test=X_test,
                treatment_plans=treatment_plans,
                verbose=False,
            )

            # Step 2: arm balancing (LR / SVR / NN only -- never FDR).
            y_tr_m = y_train
            if config.should_balance(name):
                balance_ratio = config.get_balance_config(data_name)
                if balance_ratio is not None:
                    X_tr_m, y_tr_m = config.balance_training_arms(
                        X_train=X_tr_m,
                        y_train=y_train,
                        treatment_plans=treatment_plans,
                        max_ratio=balance_ratio,
                        seed=config.SEED,
                    )

            rec_df = recommend_fn(X_tr_m, y_tr_m, X_te_m, treatment_plans)

            combined_df, _, _ = compare_recommendations(
                rec_df, test_data, outcome_col, treatment_plans,
                propensity=propensity_test_ood,
            )
            combined_df.to_csv(
                os.path.join(output_path, f"{data_name}_{name}_REC.csv"), index=False
            )
            evaluation.save_summary_stats(summary_stats, name, 1, combined_df, outcome_col)
            print(f"    [{name}] done.")

        except Exception as e:
            print(f"    [WARN] Method '{name}' failed: {e}")

    return {"output_dir": output_path}


# ============================================================
# Top-level driver: run every dataset in its configured scenario
# ============================================================

def run_all_experiments(
    datasets: list = None,
    methods: dict = None,
    k: int = 5,
    n_repeats: int = 10,
    repeat_seeds: list = None,
    base_path: str = None,
):
    """Runs FDR + all baselines for every dataset, dispatching to the CV or
    OOD driver according to each dataset's configured scenario."""
    base_path = base_path or os.getcwd()
    dataset_names = datasets or list(config.DATASET_REGISTRY.keys())

    for data_name in dataset_names:
        cfg = config.DATASET_REGISTRY[data_name]
        print(f"\nRunning experiments for dataset: {data_name}  (scenario={cfg['scenario']})")

        if cfg["scenario"] == "cv":
            run_cv_pipeline_entry(
                data_name=data_name,
                treatment_plans=cfg["treatment_plans"],
                outcome_col=cfg["outcome_col"],
                remove_cols=cfg["remove_cols"],
                base_path=base_path,
                methods=methods,
                k=k,
                n_repeats=n_repeats,
                repeat_seeds=repeat_seeds,
            )
        elif cfg["scenario"] == "ood":
            run_ood_pipeline_entry(
                data_name=data_name,
                treatment_plans=cfg["treatment_plans"],
                outcome_col=cfg["outcome_col"],
                remove_cols=cfg["remove_cols"],
                base_path=base_path,
                methods=methods,
            )
        else:
            raise ValueError(f"Unknown scenario '{cfg['scenario']}' for dataset '{data_name}'")


# ============================================================
# CLI entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Run FDR + baselines across all scenarios, then evaluate.")
    parser.add_argument("--datasets", nargs="*", default=None,
                         help="Subset of dataset names to run (default: all in config.DATASET_REGISTRY).")
    parser.add_argument("--methods", nargs="*", default=None,
                         help="Subset of method names to run (default: FDR + all baselines).")
    parser.add_argument("--k", type=int, default=5, help="Number of CV folds (CV scenario only).")
    parser.add_argument("--n-repeats", type=int, default=10, help="Number of repeated CV runs / stratified resamples (CV scenario only).")
    parser.add_argument("--no-run", action="store_true", help="Skip running methods; evaluate existing output/ only.")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation; only run methods.")
    parser.add_argument("--base-path", default=None, help="Base path containing input/ and output/ (default: cwd).")
    args = parser.parse_args()

    base_path = args.base_path or os.getcwd()

    methods = None
    if args.methods:
        all_methods = get_all_methods()
        methods = {m: all_methods[m] for m in args.methods if m in all_methods}
        missing = [m for m in args.methods if m not in all_methods]
        if missing:
            print(f"  [WARN] Unknown methods ignored: {missing}")

    if not args.no_run:
        run_all_experiments(
            datasets=args.datasets,
            methods=methods,
            k=args.k,
            n_repeats=args.n_repeats,
            base_path=base_path,
        )

    if not args.no_eval:
        output_folder = os.path.join(base_path, evaluation.OUTPUT_FOLDER_NAME)
        config.ensure_dir(output_folder)
        evaluation.evaluate_all_datasets(output_folder)


if __name__ == "__main__":
    main()
