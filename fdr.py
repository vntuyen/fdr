# -*- coding: utf-8 -*-
"""
fdr.py
======
FDR: the proposed method. There is only ONE FDR method -- a T-Learner over
TabPFN (one TabPFN regressor fit per treatment arm, predicting on held-out
patients with no treatment-column toggling).

Two SCENARIOS adapt how FDR is *evaluated*, not what FDR *is*:

  - "cv"  (recommend_fdr_cv):   used inside repeated k-fold cross-validation.
          Folds are small-ish and repeated many times, so FDR is fit ONCE per
          fold (no inner ensembling) to keep runtime tractable across the
          outer n_repeats x k_folds grid; run-to-run variance is already
          captured by the repeated-CV seeds in run_experiments.py.

  - "ood" (recommend_fdr_ood):  used for the fixed train/test OOD split,
          which is run only once per dataset (no outer repeats). Here FDR
          additionally:
            (a) selects the top-K non-treatment features by F-score (the
                OOD train cohort can have many more candidate columns, and
                feature selection improves robustness under distribution
                shift), and
            (b) averages predictions over a small seed ensemble (since there
                is no outer repeated-run loop to supply run-to-run variance).

Both share the same underlying T-Learner mechanism: train one TabPFN per
treatment arm on patients who actually received that arm, then predict the
counterfactual outcome for every test patient under every arm and recommend
the arm with the lowest predicted outcome (lower RCB = better response).

WHY A T-LEARNER (not a joint model with TP-column toggling)
-------------------------------------------------------------
A joint model (one TabPFN, treatment columns as input features) must be
asked, at inference, to flip a patient's TP indicator: e.g. a TP2 patient
queried with TP2=0, TP1=1. If no training patient ever had that exact
TP-column pattern, this is an out-of-distribution input for an in-context
learner like TabPFN, and its attention mechanism has no analogous context
to retrieve -- producing erratic counterfactual outputs (observed as
negative recovery uplift on severely imbalanced arms such as multi_ARTemis,
TP2=45 vs TP3=3).

The T-Learner sidesteps this entirely: each arm's model is trained only on
patients from that arm and never sees a toggled TP column. At inference it
predicts the outcome for ALL test patients from their (non-TP) features
alone. Arms with too few training samples fall back to the training-set
mean, a safe neutral estimate. No arm balancing is needed or applied for
FDR -- each arm model only ever sees its own arm's data, so majority-arm
dominance cannot bias it (unlike the joint linear/NN baselines, which DO
need balance_training_arms(); see config.py).
"""

import numpy as np
import pandas as pd
from tabpfn import TabPFNRegressor

import config

# Minimum training-arm samples required to fit a reliable arm-specific
# TabPFN model. Arms below this threshold fall back to the training-set mean.
MIN_ARM_SAMPLES = 5

# ---- OOD-only settings (see module docstring) ----
OOD_ENSEMBLE_SEEDS = config.REPEAT_SEEDS  # reuse the same 10 seeds as repeated-CV
OOD_MAX_FEATURES = 40


def _make_tabpfn(seed: int = None) -> TabPFNRegressor:
    return TabPFNRegressor(
        model_path=config.TABPFN_CKPT_PATH,
        device=config.TABPFN_DEVICE,
        ignore_pretraining_limits=True,
        random_state=seed if seed is not None else config.SEED,
    )


def _select_features_for_tabpfn(X_train, y_train, treatment_plans, max_non_tp=OOD_MAX_FEATURES):
    """Select top non-TP features by F-score; always retain TP columns."""
    from sklearn.feature_selection import f_regression
    non_tp = [c for c in X_train.columns if c not in treatment_plans]
    tp_cols = [c for c in X_train.columns if c in treatment_plans]
    if len(non_tp) <= max_non_tp:
        return list(X_train.columns)
    f_scores, _ = f_regression(X_train[non_tp].fillna(0), y_train)
    top_idx = np.argsort(f_scores)[::-1][:max_non_tp]
    return [non_tp[i] for i in sorted(top_idx)] + tp_cols


# ============================================================
# Scenario 1: CV -- single fit per fold, T-Learner
# ============================================================

def recommend_fdr_cv(X_train, y_train, X_test, treatment_plans):
    """
    FDR for the repeated k-fold CV scenario: one TabPFN per treatment arm,
    fit once per fold. Run-to-run variance comes from the outer repeated-CV
    seed loop in run_experiments.py, so no inner ensembling is needed here.
    """
    feature_cols = [c for c in X_train.columns if c not in treatment_plans]
    predicted_outcomes = pd.DataFrame(index=X_test.index)

    for tp in treatment_plans:
        mask = X_train[tp] == 1
        n_arm = int(mask.sum())
        if n_arm < MIN_ARM_SAMPLES:
            predicted_outcomes[tp] = float(y_train.mean())
            print(f"      [FDR/cv] {tp}: {n_arm} train samples < {MIN_ARM_SAMPLES} → mean fallback")
            continue
        m = _make_tabpfn()
        m.fit(
            X_train.loc[mask, feature_cols].values,
            y_train[mask].values,
        )
        predicted_outcomes[tp] = m.predict(X_test[feature_cols].values)

    predicted_outcomes["REC_TP"] = predicted_outcomes.idxmin(axis=1)
    return predicted_outcomes


# ============================================================
# Scenario 2: OOD -- feature-selected, multi-seed ensembled T-Learner
# ============================================================

def recommend_fdr_ood(X_train, y_train, X_test, treatment_plans):
    """
    FDR for the fixed train/test OOD scenario: same T-Learner mechanism as
    the CV scenario, plus (a) F-score feature selection to improve
    robustness under distribution shift, and (b) averaging over a small
    seed ensemble, since the OOD evaluation runs only once (no outer
    repeated-run loop to otherwise supply run-to-run variance).
    """
    selected_cols = _select_features_for_tabpfn(X_train, y_train, treatment_plans)
    X_tr_sel = X_train[selected_cols]
    X_te_sel = X_test[selected_cols]
    feature_cols = [c for c in selected_cols if c not in treatment_plans]

    sum_preds = {tp: np.zeros(len(X_te_sel)) for tp in treatment_plans}
    n_used = {tp: 0 for tp in treatment_plans}

    for seed in OOD_ENSEMBLE_SEEDS:
        for tp in treatment_plans:
            mask = X_tr_sel[tp] == 1
            n_arm = int(mask.sum())
            if n_arm < MIN_ARM_SAMPLES:
                sum_preds[tp] += float(y_train.mean())
                n_used[tp] += 1
                continue
            m = _make_tabpfn(seed=seed)
            m.fit(
                X_tr_sel.loc[mask, feature_cols].values,
                y_train[mask].values,
            )
            sum_preds[tp] += m.predict(X_te_sel[feature_cols].values)
            n_used[tp] += 1

    predicted_outcomes = pd.DataFrame(index=X_test.index)
    for tp in treatment_plans:
        n = max(n_used[tp], 1)
        predicted_outcomes[tp] = sum_preds[tp] / n

    predicted_outcomes["REC_TP"] = predicted_outcomes.idxmin(axis=1)
    return predicted_outcomes


def get_fdr_fn(scenario: str):
    """Return the appropriate FDR recommend_fn for a dataset's scenario."""
    if scenario == "ood":
        return recommend_fdr_ood
    return recommend_fdr_cv
