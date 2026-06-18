# FDR: Foundation-model-based Therapy Recommendation for Neoadjuvant Breast Cancer

This repository contains the code used to train and evaluate **FDR**, a TabPFN-based
recommendation method for selecting a personalised neoadjuvant therapy plan for
breast cancer patients, together with 14 baseline recommendation methods spanning
classical regression, causal meta-learners, causal forests, and more recent approaches.

## Repository contents

| File | Purpose |
|---|---|
| `pipeline.py` | Within-distribution evaluation: 5-fold cross-validation (stratified by treatment arm) across 8 clinical/multi-omics datasets, for all 15 methods (FDR + 14 baselines). |
| `pipeline_ood.py` | Out-of-distribution (OOD) evaluation: models are trained once on the TransNEO multi-omics cohort and evaluated, without retraining, on the independent ARTemis multi-omics cohort. |
| `models_eval.py` | Post-hoc evaluation: reads the per-method prediction files produced by the two scripts above, computes Coverage-Adjusted Uplift (CAU) and Recovery Ratio (RR), and generates the recovery and RCB-score comparison plots plus aggregated pivot tables across all 9 datasets. |
| `preprocessing.py` | Used for FDR/LR/SVR/NN-specific feature pruning and treatment-arm balancing. |

## Datasets

| Folder | Cohort | Modality | Source |
|---|---|---|---|
| `clin_TransNEO` | TransNEO trial | Clinical only | Sammut et al., *Nature* 2022, [10.1038/s41586-021-04278-5](https://doi.org/10.1038/s41586-021-04278-5) |
| `multi_TransNEO` | TransNEO trial | Clinical + digital pathology + genomic + transcriptomic | same as above |
| `clin_ARTemis` | ARTemis / PBCP | Clinical only | Earl et al., *Lancet Oncol* 2015, [10.1016/S1470-2045(15)70137-3](https://doi.org/10.1016/S1470-2045(15)70137-3); multi-omics profiling described in Sammut et al. 2022 (above) |
| `multi_ARTemis` | ARTemis / PBCP | Clinical + digital pathology + genomic + transcriptomic | same as above |
| `multi_Trans_ART` | Combined TransNEO + ARTemis | Multi-omics (combined CV setting) | derived from the two sources above |
| `OOD_multi_Trans_ART` | TransNEO (train) -> ARTemis (test) | Multi-omics (out-of-distribution setting) | derived from the two sources above |
| `GSE41998` | AC followed by ixabepilone or paclitaxel trial | Gene expression + clinical | Horak et al., *Clin Cancer Res* 2013, [10.1158/1078-0432.CCR-12-1359](https://doi.org/10.1158/1078-0432.CCR-12-1359); GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE41998 |
| `GSE22226` | I-SPY 1 trial | Gene expression + clinical | Esserman et al., *J Clin Oncol* 2012, [10.1200/JCO.2011.39.2779](https://doi.org/10.1200/JCO.2011.39.2779); GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE22226 |
| `GSE22358` | Docetaxel-capecitabine $\pm$ trastuzumab (XeNA) trial | Gene expression + clinical | Glück et al., *Breast Cancer Res Treat* 2012, [10.1007/s10549-011-1412-7](https://doi.org/10.1007/s10549-011-1412-7); GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE22358 |

Raw data files themselves are not redistributed here -- TransNEO/ARTemis access is
governed by the original studies, and the GSE accessions above can be downloaded
directly from GEO.

## Environment

### Requirements

* Python 3.10
* CUDA 12.1+ (GPU strongly recommended; CPU fallback available but slow)
* TabPFN v3: please download `tabpfn-v3-regressor-v3_default.ckpt` from
  [Prior-Labs/tabpfn_3](https://huggingface.co/Prior-Labs/tabpfn_3/tree/main) and place it in the
  `/tabpfn` folder.

```bash
# 1. Clone the repository
git clone https://github.com/vntuyen/FDR.git
cd FDR

# 2. Create virtual environment
python3.10 -m venv fdr_env
source fdr_env/bin/activate          # Linux / macOS
# fdr_env\Scripts\activate           # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```



> **Before running:** `pipeline.py` and `pipeline_ood.py` currently hardcode the
> checkpoint path as an absolute HPC scratch path (`TABPFN_CKPT_PATH = "/scratch/..."`).
> Update that constant near the top of both files to match the `/tabpfn` folder
> convention above, e.g.
> `TABPFN_CKPT_PATH = os.path.join(os.path.dirname(__file__), "tabpfn", "tabpfn-v3-regressor-v3_default.ckpt")`,
> before running either script.

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



## Results

Coverage-Adjusted Uplift (CAU, in percentage points) for all methods across all
nine datasets. **Bold** marks the highest CAU in each row.

| Dataset | FDR | CB | NN | LR | RF | SVR | XGB | S_L | X_L | DR_L | R_L | CF | CUTS | EP_L | BITES |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GSE41998 | **17.8** | 17.5 | 1.4 | 0.7 | 4.2 | 10.8 | -0.7 | -0.9 | -3.4 | 3.7 | 3.1 | -0.1 | 3.5 | 4.6 | 3.0 |
| GSE22226 | **21.7** | 21.0 | 7.0 | 0.5 | -7.2 | 19.3 | -0.2 | -2.2 | 13.5 | 3.3 | 2.7 | 1.1 | 3.5 | 10.7 | 0.8 |
| GSE22358 | 5.9 | 3.3 | 5.4 | 5.9 | 2.7 | **11.0** | -20.6 | -3.3 | 3.1 | 3.6 | 6.1 | -20.6 | 1.3 | 4.4 | 4.6 |
| TransNEO clinical | 1.6 | 2.0 | -1.5 | -2.7 | 2.6 | 0.6 | 4.3 | -0.1 | 3.3 | -1.4 | -8.2 | **5.4** | 1.1 | 3.5 | 1.2 |
| TransNEO multi-omics | 8.0 | 10.4 | 2.4 | 2.9 | 0.9 | 3.0 | 1.9 | 2.9 | 2.8 | 2.1 | -0.8 | 4.0 | 9.5 | **13.0** | 2.0 |
| ARTemis clinical | **7.7** | 6.8 | 0.6 | 2.5 | 3.7 | 3.4 | 6.8 | -3.5 | 1.3 | 1.3 | 3.5 | 1.3 | -6.6 | -2.0 | 0.0 |
| ARTemis multi-omics | 9.4 | 8.6 | 1.8 | 4.8 | -1.4 | 2.7 | 7.9 | 3.2 | 7.5 | -15.3 | -9.6 | -3.2 | 4.8 | **11.3** | 1.4 |
| Combined TransNEO+ARTemis (CV) | **12.2** | 6.6 | -4.1 | -3.9 | 3.6 | 5.5 | 5.4 | 4.7 | 2.6 | 1.0 | -1.8 | 0.4 | 4.5 | 3.0 | 1.3 |
| TransNEO -> ARTemis (OOD) | **13.6** | 4.8 | -3.1 | -21.8 | -2.4 | 8.2 | 8.9 | 8.2 | 5.9 | 2.3 | -4.0 | 8.5 | 1.4 | 5.6 | 3.7 |
| **AVERAGE** | **10.9** | 9.0 | 1.1 | -1.2 | 0.7 | 7.1 | 1.5 | 1.0 | 4.1 | 0.1 | -1.0 | -0.4 | 2.6 | 6.0 | 2.0 |

FDR achieves the highest CAU on 5 of the 9 datasets and the highest average CAU
overall. EP-Learner outperforms FDR specifically on the standalone TransNEO and
ARTemis multi-omics cohorts; SVR leads on GSE22358; Causal Forest leads on TransNEO
clinical. See the paper for full discussion.

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
[Tuyen Vu, FDR: Foundation-model-based Therapy Recommendation for Neoadjuvant Breast Cancer, 2026 ]
```


