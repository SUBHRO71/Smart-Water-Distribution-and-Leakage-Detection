# Smart Water Leakage Detection: Five-Hybrid Comparative Study

**Revised scope (2026-09-11):** a CSV-based research pipeline comparing all five hybrids
from the problem statement. Deliver experiments, diagnostic plots and comparison tables.
No dashboard, full-network graph application, hydraulic control system or deployment.

**Status:** download/preparation and a seasonal flow baseline are committed. Additional
local Phase 1 audit, event, target and split work exists but was uncommitted at review.
SMOTE, cross-validation and the five models are planned, not implemented by this update.
Read the [existing-work review](docs/PLAN_CHANGE_REVIEW.md) before training.

## Required workflow

```text
Load CSV dataset
  ↓
Preprocessing: schema, timestamps, units, missingness audit
  ↓
Data leakage checks: target leakage • duplicates • suspicious features
  ↓
Lock train / validation / test (or train / test with internal validation)
  ↓
10-fold chronological cross-validation within training/development
  ↓  Inside EACH fold:
Fit imputation / scaling / feature selection on fold-training only
  ↓
SMOTE on fold-training only
  ↓
Initial training of all five models
  ↓
Overfitting / underfitting diagnosis
  ↓
Document correction → retrain and compare validation results
  ↓
Freeze model, preprocessing, calibration and thresholds
  ↓
Final untouched test evaluation and comparison report
```

The preprocessing stage before splitting performs audits and deterministic conversions.
Learned transformations and resampling occur inside each training fold. Use time-ordered
CV for these time series, rather than shuffled row-wise K-fold. Validation/test retain
their original distribution. [TimeSeriesSplit documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html),
[imbalanced-learn leakage guidance](https://imbalanced-learn.org/stable/common_pitfalls.html).

## All five algorithms are required

| ID | Hybrid | Bounded implementation |
| --- | --- | --- |
| M1 | 1D-CNN + LSTM | Two temporal convolutions → unidirectional LSTM → head |
| M2 | CNN + BiLSTM + Attention | CNN → historical-window BiLSTM → attention pooling → head |
| M3 | Autoencoder + LSTM | Training-fitted encoder → latent historical sequence → LSTM → head |
| M4 | Transformer + CNN | Positional encoding → Transformer → temporal CNN → pooling/head |
| M5 | GNN + LSTM | Small fixed sensor adjacency → graph convolution per time step → LSTM → head |

M5 still needs edges. Use a compact adjacency CSV derived from fold-training sensor
correlations or supplied fixed metadata; document the rule. This is a sensor-relation
model, not a full pipe-network/localization implementation. With no adjacency at all,
it cannot honestly be called a GNN.

Classification is the main comparison. Demand forecasting remains a separate output
for the original problem, evaluated with regression metrics and real-data training
batches. A shared dual head is optional, not mandatory for every model. SMOTE does not
balance continuous demand targets.

## Dataset and paper matching

Retain [BattLeDIM / L-Town](https://doi.org/10.5281/zenodo.4017659) as the starting
dataset; the full 2018 release is local. [LeakDB](https://github.com/KIOS-Research/LeakDB)
is a candidate if more independent labeled scenarios are required.

Begin with a pressure-only feature track using the same ordered channels for all models.
An expanded pressure/flow/AMR/level track may be compared separately. Do not attach
unrelated BWDF weather to L-Town. Freeze features, windows and targets before comparisons.

The [paper register](docs/PAPERS_AND_REPRODUCTION.md) lists verified publications,
their data/features and access limitations. Maintain two distinct result tracks:

1. **Paper reproduction:** exact release/scenarios, channels, transformations, windows,
   labels, training protocol, splits and metric definitions.
2. **Our controlled comparison:** all five models share the same dataset/features/targets,
   chronological folds and training-only SMOTE. Record differences from the paper.

No single verified paper was found that provides the exact five hybrids and this whole
workflow on identical public CSV features. Some L-Town papers train on newly simulated
data, rather than the downloaded CSVs. Same network name does not establish reproduction.

## Resolve target and fold feasibility first

The aggregate leak label is positive 97.8% of 2018. Existing area labels also become
constant in October–December: A/C always positive, B always negative. Fourteen observed
events do not automatically support ten independent event folds. SMOTE cannot repair
missing validation classes or create independent incidents.

Revisit pipe/event/scenario targets against the chosen paper. Use binary leak/no-leak
when only those labels exist. Normal / Leakage / Abnormal consumption needs an actual
third-class label source, with simultaneous and unknown events handled explicitly.

Produce a fold-feasibility table before training. A fold missing classes, sufficient
SMOTE neighbors or usable purged events is unsupported. Do not shuffle or duplicate
incidents to force ten valid folds; document a suitable scenario dataset or an explicit
change in fold count.

## Diagnostics and comparison metrics

- Classification: accuracy, balanced accuracy, precision, recall, macro/per-class F1,
  confusion matrices, average precision (PR-AUC definition) and ROC-AUC where defined.
- Detection: event recall, false alert events/day, median/p90 detection delay and misses.
- Risk scores: Brier score and reliability bins on real held-out data.
- Demand forecasts: MAE, RMSE and bias in physical units, in a separate table.
- Experimental comparison: per-fold scores, mean/std with valid-fold counts, class/event
  support, at least three seeds for final neural comparisons, parameters and runtime.

Use training/validation curves to diagnose fitting. Test capacity reduction, dropout,
weight decay or early stopping for overfitting; investigate labels, scaling and
optimization before increasing capacity/context for underfitting. Record corrections
and retrain using the same selection protocol. Never correct a model based on final-test
outcomes. Include no-SMOTE controls; balancing is not guaranteed to improve results.

## Existing quick-start commands

Python 3.11+; PowerShell from this repository:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\smart-water.exe download --full
.\.venv\Scripts\smart-water.exe prepare
.\.venv\Scripts\smart-water.exe baseline
.\.venv\Scripts\python.exe -m ruff check .
# Avoid the existing fixed pytest_tmp ownership conflict:
$reviewTemp = Join-Path $env:TEMP ('water-tests-' + [guid]::NewGuid().ToString('N'))
.\.venv\Scripts\python.exe -m pytest -q --basetemp $reviewTemp
```

The p227 seasonal flow score is not a leak-classification result. Raw data, processed
arrays and large model files stay ignored by Git. Save exact dependency versions and
source checksums for each run.

## Next-model instructions

Follow [HANDOFF_README.md](docs/HANDOFF_README.md) for execution and test conditions.
Read [PLAN_CHANGE_REVIEW.md](docs/PLAN_CHANGE_REVIEW.md) for reusable work and required
corrections. Earlier execution logs are historical evidence, not approval of the revised
training protocol.

BattLeDIM attribution: Vrachimis et al., 2020,
[DOI 10.5281/zenodo.4017659](https://doi.org/10.5281/zenodo.4017659), CC BY 4.0.
Other datasets retain their own terms. No software license has been selected.
