# FDR: Foundation-model-based Therapy Recommendation for Neoadjuvant Breast Cancer

This repository contains the code used to train and evaluate **FDR**, a TabPFN-based
recommendation method for selecting a personalised neoadjuvant therapy plan for
breast cancer patients, together with 14 baseline recommendation methods spanning
classical regression, causal meta-learners, causal forests, and more recent
representation/uncertainty-based approaches.

## Repository contents

| File | Purpose |
|---|---|
| `pipeline.py` | Within-distribution evaluation: 5-fold cross-validation (stratified by treatment arm) across 8 clinical/multi-omics datasets, for all 15 methods (FDR + 14 baselines). |
| `pipeline_ood.py` | Out-of-distribution (OOD) evaluation: models are trained once on the TransNEO multi-omics cohort and evaluated, without retraining, on the independent ARTemis multi-omics cohort. |
| `models_eval.py` | Post-hoc evaluation: reads the per-method prediction files produced by the two scripts above, computes Coverage-Adjusted Uplift (CAU) and Recovery Ratio (RR), and generates the recovery and RCB-score comparison plots plus aggregated pivot tables across all 9 datasets. |
| `preprocessing.py` | **Not included in this upload.** `pipeline.py` imports `preprocess_df_for_model`, `should_preprocess`, `balance_training_arms`, `get_balance_config`, and `should_balance` from this module (used for FDR/LR/SVR/NN-specific feature pruning and treatment-arm balancing). Add it to the repo before `pipeline.py` will run. |

## Requirements

```
python >= 3.9
numpy
pandas
torch
scikit-learn
catboost
xgboost
tabpfn
econml
matplotlib
```

```bash
pip install numpy pandas torch scikit-learn catboost xgboost tabpfn econml matplotlib
```

`econml` is only needed for the Causal Forest (`CF`) baseline; it is imported lazily
inside that function rather than at the top of the file.

## TabPFN checkpoint

Both `pipeline.py` and `pipeline_ood.py` currently hardcode the checkpoint path as:

```python
TABPFN_CKPT_PATH = "/scratch/sq95/tv9849/FDR/tabpfn/tabpfn-v3-regressor-v3_default.ckpt"
```

This is an HPC scratch path from development and will not exist on a fresh clone.
**Edit this constant near the top of both files** to point at wherever you store the
checkpoint locally. No internet access is required or used at run time -- both scripts
explicitly disable HuggingFace/TabPFN telemetry before import.

## Expected data layout

```
input/
  GSE41998/GSE41998.csv
  GSE22226/GSE22226.csv
  GSE22358/GSE22358.csv
  clin_TransNEO/clin_TransNEO.csv
  clin_ARTemis/clin_ARTemis.csv
  multi_TransNEO/multi_TransNEO.csv
  multi_ARTemis/multi_ARTemis.csv
  multi_Trans_ART/multi_Trans_ART.csv
  OOD_multi_Trans_ART/
    OOD_multi_Trans_ART_train.csv
    OOD_multi_Trans_ART_test.csv
```

`output/` is created automatically by the scripts; it does not need to exist beforehand.

## How to run

```bash
# 1. Within-distribution 5-fold CV on the 8 clinical / multi-omics datasets
python pipeline.py

# 2. Out-of-distribution evaluation (train on TransNEO, test on ARTemis)
python pipeline_ood.py

# 3. Compute CAU / Recovery Ratio and generate comparison plots for all 9 datasets
python models_eval.py
```

Each script can take a while to run (TabPFN and the neural baselines are the slowest
methods) -- progress is printed to stdout per dataset and per method.

## Methods evaluated

| Code | Method | Family |
|---|---|---|
| `FDR` | Proposed method (TabPFN-based) | Proposed |
| `CB` | CatBoost | Classical regression |
| `NN` | Neural Network (MLPRegressor) | Classical regression |
| `LR` | Linear Regression | Classical regression |
| `RF` | Random Forest | Classical regression |
| `SVR` | Support Vector Regression | Classical regression |
| `XGB` | XGBoost | Classical regression |
| `S_L` | S-Learner | Meta-learner |
| `X_L` | X-Learner | Meta-learner |
| `DR_L` | DR-Learner | Meta-learner |
| `R_L` | R-Learner | Meta-learner |
| `CF` | Causal Forest (EconML `CausalForestDML`) | Causal forest |
| `CUTS` | Uncertainty-conservative arm selection | Modern baseline |
| `EP_L` | EP-Learner | Modern baseline |
| `BITES` | Shared representation network with MMD arm-balancing | Modern baseline |

`FDR` is implemented two different ways depending on the script: in `pipeline.py`
it is a T-learner (one TabPFN model fit independently per treatment arm); in
`pipeline_ood.py` it is a 5-seed ensemble of a single jointly-trained TabPFN with
top-40 feature selection and treatment-column toggling at inference. See the
comments at the top of each `recommend_fdr_*`-style function for details.

## Evaluation metrics

- **Coverage-Adjusted Uplift (CAU)** -- the primary metric, normalising the
  Following-vs-Not-Following recovery uplift by what fraction of patients the
  policy actually covers. Computed and pivoted across datasets in
  `output/output_AllDatasets_CAU_Pivot.csv` / `..._CAU_pp_Pivot.csv`.
- **Recovery Ratio (RR)** -- a simpler, coverage-unadjusted secondary measure
  (Following recovery rate / Not-Following recovery rate), saved per dataset
  (`<dataset>_Recovery_Ratio.csv`) and pivoted across all datasets in
  `output/output_AllDatasets_Recovery_Ratio_Pivot.csv`.
- **Average RCB Score comparison** -- mean RCB score (with standard error) in the
  Following vs. Not Following subgroups, saved per dataset
  (`<dataset>_RCB_Score_Comparison.csv`) and plotted (`<dataset>_RCB_Score_comparison.png`).

Note: pCR-based recovery outputs (recovery counts, Recovery Ratio, recovery plot) are
not produced for `GSE22358`, since our cleaned subset of that cohort does not retain a
usable pCR column (`has_pcr = False` in the dataset registry); RCB-score outputs are
still produced for it.

## Output files (per dataset, under `output/<dataset>/`)

- `<dataset>_<method>_REC.csv` -- per-patient predictions, recommended plan, and
  Following/Not-Following label for each method.
- `<dataset>_model_performance.csv` -- MSE / MAE / R2 for the standard regression
  models (causal-style methods are excluded, since they do not expose a plain
  `.predict()`).
- `<dataset>_Recovery_Ratio.csv`, `<dataset>_RCB_Score_Comparison.csv` -- per-method
  subgroup summary statistics.
- `<dataset>_recovery_comparison.png`, `<dataset>_RCB_Score_comparison.png` -- the
  two comparison figures described above.

Aggregated, cross-dataset pivot tables are written directly under `output/`:
`output_AllDatasets_Recovery_Ratio.csv`, `output_AllDatasets_Metrics_Long.csv`, and
the four `output_AllDatasets_*_Pivot.csv` files (Recovery Ratio, CAU, CAU in
percentage points, and extra recoveries per 100 patients).

## Citation

If you use this code, please cite:

```
[Author names, paper title, venue, year -- fill in once finalised]
```

## License

[Add a license file, e.g. MIT or Apache-2.0, before publishing]
