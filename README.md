# Smart Water Leakage Detection: Research and Execution Guide

Updated: 2026-09-12. This is the single implementation roadmap and phase handoff.

## Exact project requirements

Build a reproducible Python CSV research pipeline for water leakage detection. Implement
all five hybrids below, study the relevant papers using their actual datasets and feature
pipelines, reproduce their documented split, and additionally evaluate with mandatory
10-fold cross-validation. Compare results, identify a supported research gap, and test a
specific improvement that addresses it. Keep code and evidence in the public GitHub repo.

The workflow is: load CSVs -> audit/preprocess -> check target, duplicate and suspicious
feature leakage -> lock splits -> fit training-only preprocessing -> training-only SMOTE
-> initial models -> diagnose overfitting/underfitting -> correct and retrain -> freeze
settings -> final test and comparison.

No dashboard, deployment or full hydraulic-network application is required. M5 still
requires a small, documented graph. Demand forecasting and abnormal consumption belong
to the original problem; handle them as separate target-supported extensions. Do not
invent an abnormal-consumption class from binary leak labels.

## Required models and paper relationships

| ID | Our required hybrid | Bounded implementation | Related paper |
| --- | --- | --- | --- |
| M1 | 1D-CNN + LSTM | Two Conv1D layers, unidirectional LSTM, classification head | P5 is related; a closer reference remains to be selected |
| M2 | CNN + BiLSTM + Attention | CNN, historical-window BiLSTM, attention pooling, head | P5 uses CNN/LSTM/attention; exact BiLSTM match unverified |
| M3 | Autoencoder + LSTM | Fold-trained encoder, latent sequence, LSTM classifier | P2 studies AE, VAE and LSTM-AE; architecture differs |
| M4 | Transformer + CNN | Positional encoding, Transformer, temporal CNN, head | P4 is a Transformer reconstruction reference |
| M5 | GNN + LSTM | Sensor graph convolution per time step, LSTM, head | P3 uses graph reconstruction/prediction; architecture differs |

P1 is the BattLeDIM benchmark paper, not a CNN-LSTM architecture reference. The
[paper register](docs/PAPERS_AND_REPRODUCTION.md) holds titles, links, recipes and access
limitations. These are candidate references, not five already reproducible experiments.

## Three experiment tracks

1. **Paper reproduction:** obtain the paper's actual release/scenarios, features, labels,
   preprocessing, architecture and split; run its published protocol once. If we retain
   a different hybrid, label that run a paper-data adaptation, not an exact reproduction.
2. **Mandatory additional CV:** run 10 folds for our applicable model on each selected
   paper dataset's training/development portion, even when the paper did not use CV.
   Keep the paper's final test set locked. If the paper already uses 10 folds, reproduce
   that protocol and explicitly document whether our added protocol differs.
3. **Controlled comparison and gap experiment:** run all five on identical suitable data,
   features, folds and metrics; compare a proposed improvement against the original
   version. LeakDB is the prepared common benchmark, not a substitute for every paper's
   original data. Scores from different datasets must not be ranked as model superiority.

Preserve the paper's train/test ratio and assignments, or its train/validation/test
protocol, as an additional required evaluation alongside CV. An 80/20 train-validation
split is not automatically an 80/20 train-test split. Verify unknown details rather than
guessing. Reproduce the paper's imbalance treatment; our SMOTE addition is a separate run
if the paper does not use it.

## Current evidence and limitations

| Component | Status |
| --- | --- |
| BattLeDIM download, sensor/event audit and boundary corrections | Implemented; historical area labels fail classification feasibility |
| LeakDB preparation | Implemented: 175,200 rows, ten scenarios, 31 selected pressure channels |
| LeakDB windowing and fold-local scaling/SMOTE | Implemented; one real-fold smoke run completed |
| Verification at previous phase boundary | 33 tests passed and Ruff passed |
| Five hybrids and common training runner | Not implemented |
| Exact paper datasets/recipes | Incomplete; see paper register |
| Paper reproduction, trained CV and final accuracy tables | Not completed |
| Research gap and novelty claim | Not yet verified or selected |

The prepared common configuration is [configs/experiment.json](configs/experiment.json).
LeakDB source commit: `131144ba423a82639f881adab0d493ab50e3b2fb`; archive
`CCWI-WDSA2018/Benchmarks/Hanoi_CMH.zip`, MD5 `700e10f8a90f028f838fcae49660225c`.
Features: pressure Node 2 through Node 32, ordered numerically; Node 1 is constant zero.
Target: leak presence at the last input timestamp. Window: 24 half-hour observations
(12-hour coverage), stride 6 (3 hours). There are 21,810 development and 7,360 test windows.

The common benchmark uses Jan-Sep 2017 development and Oct-Dec locked test, with ten
leave-one-scenario-out folds inside development. This grouped CV is distinct from
expanding chronological CV. All training folds have both classes; six assessment
scenarios have no positive development labels. Report aggregate out-of-fold scores and
undefined per-fold metrics explicitly. Counts alone do not prove scenario independence:
audit shared simulation sources and duplicate windows before training. Three-hour
sampling limits detection-delay resolution; freeze a denser protocol before experiments
if rapid alerts are the gap being studied. Test windows currently allow observed past
context; report this operational protocol and boundary-carryover events explicitly.

The [feasibility report](docs/results/prerequisite-and-feasibility.md) records the dataset
decision and smoke counts. Existing `configs/targets.json` and `configs/splits.json`
describe the BattLeDIM audit, not the accepted five-model classification protocol.

## Exact steps and acceptance conditions

### Phase 1: Freeze literature, data and the research question

1. Read each selected paper's full method, experiments, limitations/future work and
   supplements. Record citations to sections/tables, original model, actual data access,
   preprocessing and split. Resolve the M1 reference and other architecture mismatches.
2. Create one paper experiment config per verified recipe under `configs/papers/`.
   Record release/checksum, scenario IDs, units, ordered features, feature equations,
   cadence, window/stride, labels, model/loss/optimizer, seeds, split IDs and metrics.
   Request or obtain author data where needed; mark unavailable experiments blocked.
3. Write `docs/results/research-gap.md`: cited limitation, what has already been tried,
   our testable question, proposed change, unchanged comparator, expected measurable
   outcome and novelty evidence. A five-model comparison alone does not establish a gap.

**Accept when:** data are actually available, every input maps to a source column or
documented transform, split details are verified, and the gap has evidence. Keep proposed
gaps distinct from confirmed novelty. This is the next phase to execute.

### Phase 2: Audit CSVs, targets and leakage

4. Load immutable raw CSVs; audit units, cadence, missing cells versus absent timestamps,
   duplicate source identities, constant channels and sensor coverage. Inventory leak
   onset/end, duration, magnitude, location mapping and censoring where labels support it.
5. Freeze the target and feature allowlist. Report timestamp prevalence and independent
   event coverage separately; unknown labels stay unknown. Count events with independent
   labels when possible, otherwise mark event coverage unknown.
6. Exclude label/repair/leak-flow metadata from predictors. Audit suspicious development
   correlations, duplicate windows and shared source simulations. Test that changing
   future observations cannot change a past feature. Use only causal imputation in the
   operational comparison; document differences from the original paper's preprocessing.

**Accept when:** prohibited inputs are absent, missingness is handled explicitly, event
support is recorded, and source identities can be tracked through every split. A constant
majority prediction must be reported to expose trivial accuracy.

### Phase 3: Freeze the paper split and mandatory ten folds

7. Materialize the paper's original split with its exact ratio/date/scenario assignments.
   Reserve final test data. Create internal training-only validation if tuning requires it.
8. Construct exactly ten additional folds inside development: grouped scenario/event
   folds for independent simulations, or expanding temporal folds for continuous series.
   Purge overlapping raw input/target intervals and shared events as the protocol requires.
   Never shuffle overlapping windows into supposedly independent folds.
9. Save fold manifests: source IDs, input/target intervals, purge reasons, class/event
   counts and SMOTE neighbor support. Reuse identical fold IDs across comparable models.

**Accept when:** all ten assignments are explicit, final test never enters tuning, each
training fold supports the chosen method, and holdout metrics state their class support.
Single-class assessment folds contribute appropriate normal-only/aggregate metrics; they
do not justify fabricated recall/AUC. If ten scientifically valid folds are impossible,
obtain suitable independent data or report a blocker; do not silently reduce the count.
Reserve inner validation for early stopping rather than using outer fold assessment.

### Phase 4: Preprocessing and balancing

10. Fit imputers, scalers and feature selectors on real fold-training data only. Refit
    them in every fold. Transform holdouts without fitting or resampling.
11. Compare no-SMOTE and SMOTE training variants. Record seed, neighbors and class counts.
    For our continuous sequence inputs, flatten whole windows, interpolate, then restore
    the original time/channel shape. Audit implausible ranges, temporal smoothing and
    source-event mixing; synthetic windows are not independent physical leak events.

**Accept when:** held-out source/label hashes and counts stay unchanged, training support
exceeds the neighbor requirement, arrays are finite and ordered, and no test sample enters
`fit` or `fit_resample`. IDs, categorical metadata and demand targets are not interpolated.
An unsupervised normal-only paper run must not be converted to supervised SMOTE training.

### Phase 5: Implement and initially train the five models

12. Build all five behind one training/prediction interface. Fit M3's encoder on real
    training data only. Estimate M5 adjacency from real fold-training correlations or
    documented fixed metadata; save sensor order, edges, self-loop and normalization rules.
    Historical BiLSTM/Transformer inputs must contain nothing after the prediction origin.
13. Verify forward/backward shapes, finite gradients, tiny-batch learning, checkpoint
    reload equivalence and CPU inference. Verify the graph is actually used.
14. Run the paper protocol and the additional ten-fold experiments as separately named
    runs. Save predictions, checkpoints, learning curves, runtime and exact package versions.
    Use equal tuning budgets and at least three seeds for final neural comparisons.

Optional conventional baselines support interpretation and never replace a required
hybrid: use a classification linear baseline/gradient boosting for leak labels; keep
ridge regression for demand and Isolation Forest as a separately specified anomaly model.

**Accept when:** all five execute, every run identifies its data/features/split/seed,
and each fold reinitializes models, transformations, encoders, adjacency and samplers.

### Phase 6: Diagnose, correct and test the gap

15. Compare real-training and inner-validation losses/metrics. Worsening validation with
    improving training suggests overfitting: test stopping, dropout, regularization or
    smaller capacity. Poor training and validation require label/scaling/optimization
    checks before increasing capacity. Investigate suspiciously perfect scores for leakage.
16. Record each correction and reason; retrain using the same selection protocol. Compare
    the proposed gap improvement to an unchanged baseline with controlled ablations.
    Report failed improvements as valid outcomes, including variation across seeds/folds.

**Accept when:** corrections use development evidence only, comparisons share scoring
rows, and ablations isolate the claimed contribution. Do not tune on final-test outcomes.

### Phase 7: Calibration, alerts and final evaluation

17. Fit sigmoid/isotonic calibration on reserved real development predictions with adequate
    class support. Choose thresholds separately from calibrator fitting; reserve independent
    assessment predictions for reporting. Declare unsupported calibration explicitly.
18. Freeze alert persistence, gap merging, cooldown and one-to-one event matching. Record
    alert time when persistence is confirmed; merging must not backdate detection. Define
    event misses, censored/carryover events and valid observation-day denominators.
19. Freeze models, preprocessing, thresholds and analysis choices; evaluate locked test
    once as the planned final evaluation. Do not correct/retrain based on its results.

**Accept when:** confusion/Brier/event metrics pass hand-calculated fixtures; single-class
and no-event cases are explicit; saved predictions reproduce every reported score.

### Phase 8: Compare, conclude and hand off

20. Publish separate tables for: paper-reported versus faithfully reproduced/adapted runs;
    all ten CV folds plus aggregate out-of-fold and mean/std with valid-fold counts; common
    final test; and gap baseline versus proposed change/ablations. Do not mix data regimes.
21. Report accuracy, balanced accuracy, precision, recall, macro/per-class F1, confusion
    matrix, average precision (define PR-AUC), ROC-AUC where supported, Brier/reliability,
    event recall, false alerts/day, median/p90 detected-event delay, misses and runtime.
    Include uncertainty and event/scenario support; overlapping windows are not independent
    evidence for statistical significance.
22. Answer the research question using the evidence and state limitations. Demand uses
    real observed targets with MAE/RMSE/bias in a separate table. Abnormal consumption
    requires independently labeled cases or clearly declared synthetic scenarios.

**Accept when:** every result traces to a configuration and prediction file; novelty is
supported, not assumed; unsupported claims are removed; commands reproduce the results.

## Existing commands (preparation only)

PowerShell, Python 3.11+:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,ml]"
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
```

If the ignored LeakDB source folder is absent, acquire the pinned source once:

```powershell
git clone https://github.com/KIOS-Research/LeakDB.git data/raw/LeakDB-source
git -C data/raw/LeakDB-source checkout --detach 131144ba423a82639f881adab0d493ab50e3b2fb
.\.venv\Scripts\smart-water.exe prepare-leakdb
.\.venv\Scripts\smart-water.exe preprocess-leakdb
```

`preprocess-leakdb` runs one SMOTE fold smoke check, not ten-fold model training. Its
defaults match the current common configuration; it does not yet load paper configs.
Inspect `smart-water --help` for retained BattLeDIM download, prepare, audit, inventory,
targets, splits, leakage-audit, cv-feasibility and seasonal-flow baseline commands. The
seasonal baseline is a preparation sanity check, not a trained leak classifier. Neural
dependencies are available through `.[deep]` when Phase 5 starts.

## Repository and phase handoff rules

- `README.md`: sole requirements, execution steps and current status.
- `docs/PAPERS_AND_REPRODUCTION.md`: cited papers and recipe/access evidence.
- `docs/datasets.md`: dataset provenance and alternatives.
- `docs/results/`: compact observed audit and research results; historical findings are
  labeled. `docs/verification.md` records the original starter environment/results.
- `src/smart_water/`, `tests/`, `configs/`: reusable implementation, checks and contracts.
- `data/`, `reports/`, `models/`: local ignored data and large outputs. Commit compact
  evidence and source checksums, not raw datasets, credentials or large checkpoints.

After each phase, run appropriate tests/lint, update this README's status and add measured
evidence under `docs/results/`. Record changes, exact commands, limitations and next step;
stop at the phase boundary for review. Do not claim future steps are implemented.

Next action: **Phase 1**, verify paper-specific datasets/recipes and select an evidenced
research gap before extending model training. Prepared LeakDB work remains reusable.
