# FDR: Foundation-Model-Based Therapy Recommendation for Neoadjuvant Breast Cancer


---

## Overview

Selecting the most appropriate neoadjuvant therapy plan for a breast cancer patient is difficult because treatment response varies widely between patients, and the available cohorts are small and high-dimensional.

**FDR** learns one TabPFN foundation-model regressor for each candidate therapy plan. It predicts the continuous Residual Cancer Burden (RCB) score each patient would have under every plan, and recommends the plan with the lowest predicted score.

FDR is compared with ten baselines on six clinical and multi-omics settings built from two independent breast cancer cohorts (TransNEO and ARTemis). The evaluation covers both repeated within-cohort cross-validation and cross-cohort evaluation without retraining.

## Key results

Primary metric: **Coverage-Adjusted Uplift (CAU)**, the gain in pathological complete response (pCR) rate among patients whose received therapy matched the recommendation, scaled by recommendation coverage. Values are in percentage points (pp), as mean ± std over five repeated runs.

| Setting | FDR | Best baseline |
|---|---|---|
| TransNEO clinical | 4.45 ± 2.08 | **5.13 ± 2.53** (X-Learner) |
| ARTemis clinical | **7.58 ± 3.13** | 4.12 ± 4.87 (X-Learner) |
| TransNEO multi-omics | **14.73 ± 3.24** | 13.34 ± 2.29 (Causal Forest) |
| ARTemis multi-omics | **14.09 ± 1.68** | 12.14 ± 2.08 (Causal Forest) |
| TransNEO + ARTemis multi-omics (CV) | **12.06 ± 1.62** | 11.79 ± 1.02 (Causal Forest) |
| TransNEO → ARTemis multi-omics (OOD) | **11.00 ± 0.00** | 8.89 ± 0.00 (XGBoost) |

- **Ranking.** FDR ranks first in 5 of 6 settings and second in the remaining one.
- **Cross-setting average.** FDR has the highest mean of all 11 methods on every metric:
  - CAU: 10.65 pp, 95% CI 7.72–13.52.
  - Recovery Rate Difference (RRD): 24.31 pp, 95% CI 17.70–29.80.
  - Recovery Ratio: 2.87, 95% CI 2.12–3.56.
- **Multi-omics.** Adding multi-omics features increases FDR's CAU in both cohorts:
  - TransNEO: 4.45 → 14.73 pp.
  - ARTemis: 7.58 → 14.09 pp.
- **Significance.** FDR's CAU exceeds chance in all 6 settings (one-sided permutation test, p < 0.05). It remains significant in 5 of 6 after Benjamini–Hochberg correction; the exception is TransNEO clinical.


## Repository structure

```
FDR/
├── config.py                  # Paths, dataset registry, method registry, run seeds
├── fdr.py                     # The proposed method 
├── baselines.py               # The ten baseline methods
├── ctr_causaltree.R           # R back-end for the CTR baseline
├── statistical_validation.py  # Bootstrap CIs, permutation tests, BH correction, IPW, E-values
├── evaluation.py              # Metrics, tables and figures used in the paper
├── run_experiments.py         # Main entry point: runs all methods, then evaluates
├── requirements.txt           # Environment installation
├── tabpfn/                    # Place the TabPFN checkpoint here (not tracked)
└── input/                     # Datasets 
```

| File | Description |
|---|---|
| `config.py` | Single source of truth: environment, CUDA and TabPFN-checkpoint setup; the six-dataset registry (`DATASET_REGISTRY`); the method registry (`METHOD_META`); and the repeated-run seeds (`REPEAT_SEEDS`). |
| `fdr.py` | The proposed method. It has two scenario-specific entry points, `recommend_fdr_cv` and `recommend_fdr_ood`, which share the same mechanism. |
| `baselines.py` | The ten baselines (see [Methods](#methods-compared)). |
| `ctr_causaltree.R` | R back-end for CTR (honest causal trees via `causalTree`), adapted from [vntuyen/ctr](https://github.com/vntuyen/ctr). It is called by `baselines.recommend_ctr`. |
| `statistical_validation.py` | Patient-level bootstrap CIs, permutation tests, Bonferroni/BH correction, covariate balance (SMD), IPW-adjusted CAU, IPW/doubly-robust policy value, and E-values. |
| `evaluation.py` | Computes CAU, RRD, Recovery Ratio and average RCB per dataset. Builds the clinical-vs-multi-omics comparison, cross-dataset summaries, the FDR significance table, and all paper figures. |
| `run_experiments.py` | Runs FDR and all baselines on every dataset in its configured scenario (CV or OOD), then runs the evaluation. |

## Installation

### Requirements

- Python 3.10
- CUDA 12.1+ (a GPU is strongly recommended; the CPU fallback works but is slow)
- R ≥ 4.3 (only needed for the CTR baseline)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/vntuyen/FDR.git
cd FDR

# 2. Create and activate a virtual environment
python3.10 -m venv fdr_env
source fdr_env/bin/activate          # Linux / macOS
# fdr_env\Scripts\activate           # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### TabPFN checkpoint

Download `tabpfn-v3-regressor-v3_default.ckpt` from [Prior-Labs/tabpfn_3](https://huggingface.co/Prior-Labs/tabpfn_3/tree/main) and place it in the `tabpfn/` folder of the repository:

```
FDR/tabpfn/tabpfn-v3-regressor-v3_default.ckpt
```

To keep the checkpoint elsewhere (for example, on an HPC scratch drive), set:

```bash
export TABPFN_CKPT_PATH=/path/to/tabpfn-v3-regressor-v3_default.ckpt
```

### CTR baseline 
CTR runs in R and is called from Python through `Rscript`. Install the R packages:

```r
install.packages(c("rpart", "rpart.plot", "remotes"))
remotes::install_github("susanathey/causalTree")
```

If `Rscript` is not on your `PATH`, set `export RSCRIPT_BIN=/path/to/Rscript`. Without R, all other methods still run; CTR is reported as failed and skipped.

## Data

All six datasets are derived from two neoadjuvant breast cancer cohorts, downloaded from their [Github repository)](https://github.com/micrisor/NAT-ML/tree/main/inputs):

- **TransNEO**: Sammut et al., *Nature*, 2022.
- **ARTemis/PBCP**: Earl et al., *Lancet Oncology*, 2015.

All datasets share a continuous outcome (`RCB.score`) and four treatment plans (`TP1`–`TP4`). Each plan is a taxane backbone with optional anthracycline and anti-HER2 add-ons.

| Dataset key | Cohort(s) | Features | Scenario | N |
|---|---|---|---|---|
| `clin_TransNEO` | TransNEO | Clinical (8) | CV | 147 |
| `clin_ARTemis` | ARTemis/PBCP | Clinical (8) | CV | 72 |
| `multi_TransNEO` | TransNEO | Multi-omics (74) | CV | 147 |
| `multi_ARTemis` | ARTemis/PBCP | Multi-omics (74) | CV | 72 |
| `multi_Trans_ART` | TransNEO + ARTemis | Multi-omics (74) | CV | 219 |
| `OOD_multi_Trans_ART` | TransNEO → ARTemis | Multi-omics (74) | OOD | 147 train / 72 test |

**Data availability.** This repository does not redistribute patient data. Please obtain the cohort data through the access routes described in the original publications. Then place the processed files as follows:

```
input/
├── clin_TransNEO/clin_TransNEO.csv
├── clin_ARTemis/clin_ARTemis.csv
├── multi_TransNEO/multi_TransNEO.csv
├── multi_ARTemis/multi_ARTemis.csv
├── multi_Trans_ART/multi_Trans_ART.csv
└── OOD_multi_Trans_ART/
    ├── OOD_multi_Trans_ART_train.csv
    └── OOD_multi_Trans_ART_test.csv
```

<!-- TODO: if a preprocessing script or data dictionary is released, link it here. -->

## Usage

```bash
# Run everything (FDR + 10 baselines on all 6 datasets), then evaluate
python run_experiments.py

# Evaluation only, reusing an existing output/ folder
python run_experiments.py --no-run

# Run only, skipping evaluation
python run_experiments.py --no-eval

# A subset of datasets or methods
python run_experiments.py --datasets multi_ARTemis OOD_multi_Trans_ART
python run_experiments.py --methods FDR CB XGB

# Number of repeated runs and CV folds
python run_experiments.py --n-repeats 5 --k 5
```

The paper reports five repeated runs of 5-fold cross-validation.
<!-- TODO: check the --n-repeats default in run_experiments.py matches the paper. -->

## Outputs

The `output/` folder is created automatically.

**Per dataset** (`output/<dataset>/`):

- `<dataset>_<method>_REC.csv` / `_REC_all_runs.csv`: per-patient recommendations for each method.
- `<dataset>_Recovery_Metrics_summary.csv`: CAU, RRD and Recovery Ratio (mean ± std across runs, with 95% bootstrap CI).
- `<dataset>_RCB_Score_Comparison.csv`: average RCB for followers vs. non-followers.
- `<dataset>_Statistical_Validation.csv`: bootstrap CIs, permutation p-values, IPW-adjusted CAU, E-values, and IPW/DR policy value.
- Recovery Ratio and RCB comparison plots.

**Across datasets** (`output/`):

| File | Contents | Paper                                     |
|---|---|-------------------------------------------|
| `output_AllDatasets_Metrics.csv` | CAU/RRD/RR for every (dataset, method): mean, across-run std, 95% CI | —                                         |
| `output_AllMethods_CrossDataset_Mean_CI95.csv` | Mean and 95% CI across the six settings, per method | Table 4: Cross-dataset table              |
| `output_AllDatasets_CAU_pp_Formatted.csv` | CAU as "mean ± std" strings, datasets × methods | Table 5: CAU table                        |
| `output_Clinical_vs_Multiomics_within_cohort.png` / `.csv` | Clinical vs. multi-omics, FDR and all-method mean, per cohort | Figure 3: Clinical vs. multi-omics figure |
| `output_StatisticalValidation_AllDatasets.csv` | All validation statistics, with Bonferroni and BH-adjusted p-values | —                                         |
| `output_FDR_Significance_Table.csv` | Per-dataset CAU CI, permutation p, BH p, E-value and IPW CAU for FDR | Table 6: Significance table               |


## Methods compared

| Method | Key | Family                          |
|---|---|---------------------------------|
| **FDR (ours)** | `FDR` | Foundation model                |
| CatBoost | `CB` | Classical ML                    |
| XGBoost | `XGB` | Classical ML                    |
| S-Learner | `S_L` | Meta-learner                    |
| X-Learner | `X_L` | Meta-learner                    |
| DR-Learner | `DR_L` | Meta-learner                    |
| R-Learner | `R_L` | Meta-learner                    |
| Causal Forest | `CF` | Causality model                 |
| CTR | `CTR` | Causality model                 |
| CUTS | `CUTS` | Recent treatment recommendation |
| BITES | `BITES` | Recent treatment recommendation |


