# FDR: Foundation-model-based Therapy Recommendation for Neoadjuvant Breast Cancer

This repository contains the code used to train and evaluate **FDR**, a TabPFN-based
recommendation method for selecting a personalised neoadjuvant therapy plan for
breast cancer patients, together with 9 baseline recommendation methods spanning
classical regression, causal meta-learners, causal forests, and recent approaches.

## Repository contents

| File | Description |
|---|---|
| `config.py` | Single source of truth: environment/CUDA/TabPFN-checkpoint setup, the 6-dataset registry (`DATASET_REGISTRY`), the 10-method registry (`METHOD_META`), and the repeated-run seed list (`REPEAT_SEEDS`). |
| `fdr.py` | The proposed method, two scenario-specific entry points, `recommend_fdr_cv` and `recommend_fdr_ood`, sharing the same underlying mechanism. |
| `baselines.py` | All nine baseline methods: CatBoost, XGBoost (classical); S-/X-/DR-/R-Learner (meta-learners); Causal Forest (causal); CUTS, BITES (modern). |
| `statistical_validation.py` | Bootstrap confidence intervals and cross-dataset stability metrics. |
| `evaluation.py` | The full evaluation pipeline: CAU / Recovery Ratio / RRD / average-RCB metrics per dataset, the paired Clinical-vs-Multi-omics comparison, the within-cohort FDR-vs-all-method figure, cross-dataset summaries with 95% bootstrap CIs, and every plot used in the paper. |
| `run_experiments.py` | Runs FDR + all baselines across every dataset in its configured scenario (CV or OOD), then calls the full evaluation suite. |

## Datasets

Six datasets are used, all drawn from the **TransNEO** (Sammut et al., *Nature*, 2022)
and **ARTemis/PBCP** (Earl et al., *Lancet Oncology*, 2015) neoadjuvant breast cancer
cohorts, sharing a common outcome (continuous `RCB.score`) and a common four-arm
treatment-plan structure (`TP1`-`TP4`: taxane backbone with binary anthracycline/anti-HER2
add-ons).

| Dataset key | Cohort(s) | Feature view | Scenario | N (true cohort) |
|---|---|---|---|---|
| `clin_TransNEO` | TransNEO | Clinical only (8 features) | CV | 147 |
| `clin_ARTemis` | ARTemis/PBCP | Clinical only (8 features) | CV | 72 |
| `multi_TransNEO` | TransNEO | Multi-omics (74 features) | CV | 147 |
| `multi_ARTemis` | ARTemis/PBCP | Multi-omics (74 features) | CV | 72 |
| `multi_Trans_ART` | TransNEO + ARTemis (combined) | Multi-omics (74 features) | CV | 219 |
| `OOD_multi_Trans_ART` | TransNEO (train) -> ARTemis (test) | Multi-omics (74 features) | OOD (fixed train/test split, different cohorts) | 147 train / 72 test |


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



> **Before running:**  The checkpoint path is currently an absolute HPC scratch path (`TABPFN_CKPT_PATH = "/scratch/..."`).
> Update that constant near the top of both files to match the `/tabpfn` folder
> convention above, e.g.
> `TABPFN_CKPT_PATH = os.path.join(os.path.dirname(__file__), "tabpfn", "tabpfn-v3-regressor-v3_default.ckpt")`,
> before running either script.

## Expected data layout

```
input/
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

Running the pipeline produces, under `<base_path>/output/`:

- `output/<dataset_name>/` -- per-method prediction files (`<dataset>_<method>_REC.csv` /
  `_REC_all_runs.csv`), per-dataset metric summaries (`Recovery_Metrics_summary.csv`,
  `PolicyValue_summary.csv`, `RCB_Score_Comparison.csv`, `CovariateBalance_SMD_summary.csv`),
  and plots (Recovery Ratio, RCB comparison, DR policy uplift).
- `output/output_AllDatasets_Metrics.csv` -- combined CAU/RRD/RR per (dataset, method) with
  mean, across-run std, and 95% bootstrap CI.
- `output/output_AllMethods_CrossDataset_Mean_CI95.csv` -- mean + 95% CI across all six
  datasets, per method, for CAU/RRD/RR.
- `output/output_Clinical_vs_Multiomics_within_cohort.png` / `.csv` -- the paired,
  within-cohort Clinical-vs-Multi-omics figure (FDR and all-method mean) for TransNEO and
  ARTemis.

## Methods evaluated

**FDR** (proposed) plus nine baselines across four families:

| Method | Key | Family |
|---|---|---|
| **FDR** | `FDR` | Proposed (TabPFN T-Learner) |
| CatBoost | `CB` | A. Classical |
| XGBoost | `XGB` | A. Classical |
| S-Learner | `S_L` | B. Meta-Learner |
| X-Learner | `X_L` | B. Meta-Learner |
| DR-Learner | `DR_L` | B. Meta-Learner |
| R-Learner | `R_L` | B. Meta-Learner |
| Causal Forest | `CF` | C. Causal |
| CUTS | `CUTS` | D. Modern SOTA  |
| BITES | `BITES` | D. Modern SOTA |

## Results

Summary of the headline findings (see the paper for full detail, confidence intervals,
and significance tests):

- **Per-dataset ranking (Coverage-Adjusted Uplift, CAU):** FDR ranks 1st on 3 of 6 datasets
  (ARTemis multi-omics, the combined TransNEO+ARTemis CV setting, and the OOD setting),
  2nd on both clinical-only datasets, and 3rd on TransNEO multi-omics -- never below 3rd.
- **Cross-dataset summary (mean ± 95% CI across all 6 datasets):** FDR attains the highest
  mean CAU (7.88 pp [4.62, 10.98]), RRD (25.12 pp [14.14, 36.24]), and Recovery Ratio
  (2.61 [1.86, 3.34]) of all ten methods. Its CI is clearly separated from zero/one and
  from R-Learner (significantly negative), but overlaps with CatBoost, XGBoost, X-Learner,
  S-Learner, and CUTS -- so FDR's advantage over those specific baselines is directionally
  consistent rather than statistically established on this evidence alone.
- **Clinical vs. multi-omics:** adding multi-omics features roughly triples FDR's CAU
  (TransNEO: 2.2 -> 8.5 pp; ARTemis: 2.7 -> 9.9 pp), with the same direction of effect,
  at smaller magnitude, across the other nine methods.
- **Out-of-distribution validation:** training on TransNEO and testing without retraining
  on the independent ARTemis cohort, FDR achieves the highest CAU of all ten methods
  (11.75 pp vs. 8.89 pp for the next-best baseline, XGBoost), consistent with its matched
  in-distribution CV result (12.27 pp) -- though S-Learner edges ahead of FDR on RRD and
  Recovery Ratio in this specific setting.
- **Statistical significance:** by permutation test (1000 permutations), FDR's CAU is
  significant at p<0.05 on 5 of 6 datasets (TransNEO clinical, p=0.13, is the marginal
  exception).

All numbers above are reproducible from `output/output_AllDatasets_Metrics.csv` and
`output/output_AllMethods_CrossDataset_Mean_CI95.csv` after running the pipeline.

## Usage

```bash
# Run everything (FDR + 9 baselines, all 6 datasets) and evaluate
python run_experiments.py

# Evaluation only, reusing existing output/
python run_experiments.py --no-run

# Run only, skip evaluation
python run_experiments.py --no-eval

# Subset of datasets / methods
python run_experiments.py --datasets multi_ARTemis OOD_multi_Trans_ART
python run_experiments.py --methods FDR CB XGB

# Fewer repeated resamples (default: 10) or folds (default: 5)
python run_experiments.py --n-repeats 5 --k 5
```



