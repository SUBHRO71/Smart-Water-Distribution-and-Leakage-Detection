# Handoff: five-hybrid CSV experiment

Updated 2026-09-11. This replaces the prior four-phase roadmap. All five hybrids in the
[README](../README.md) are required candidates. No dashboard or full-network application.
This is a plan; acceptance conditions below have not automatically been met.

## 0. Review prerequisites

Read [PLAN_CHANGE_REVIEW.md](PLAN_CHANGE_REVIEW.md). Preserve local audit/event work;
fix the listed split-boundary and unknown-label issues before training. Existing frozen
targets/splits and the old execution log's gate decision need revalidation.

**Deliver:** correction record and regenerated manifests.
**Tests:** targets exactly on an exclusive partition boundary are rejected; unknown
labels remain unknown; regenerated counts agree with valid windows. Existing tests/lint pass.

## 1. Freeze a paper, data and feature contract

Use [PAPERS_AND_REPRODUCTION.md](PAPERS_AND_REPRODUCTION.md). Record exact release/scenarios,
training source, sensor names/order, units, cadence, transforms, sequence length/stride,
labels, split, architecture, optimization and metric definitions. Cite sections/tables/code;
mark unknown details rather than guessing from an abstract.

**Deliver:** proposed `configs/experiment.json` and `docs/results/reproduction-contract.md`.
**Tests:** every feature maps to an actual input column; exact training data are available
or reproduction is explicitly blocked. Paper reproduction and our SMOTE/CV comparison
have separate configurations and result tables. No incompatible cross-paper accuracy ranking.

## 2. Load CSVs and audit preprocessing

Reuse downloads/checksums and existing coverage/inventory utilities. Validate schemas,
timestamps, cadence, units, duplicate identities, missingness and sensor mapping.
Keep leak labels separate. Specify causal imputation/scaling without fitting on all data.

**Deliver:** coverage report, numerical feature allowlist and dictionary.
**Tests:** missing values, absent timestamps and duplicates are distinguished; unit
conversion is hand-checked; raw data are unchanged. Learned transforms are fitted later
inside fold-training; future values cannot affect a past feature.

## 3. Check leakage and label feasibility

Audit direct target/repair/leak-flow predictors, full-year statistics, suspicious near-perfect
features, duplicate source records, duplicate windows and shared observations across folds.
Inspect suspicious feature correlations using development only. Equal readings at different
valid times are not automatically duplicate records. Preserve source identity/provenance.

Count independent events and class support per target and interval. Existing area labels
are constant in validation, so reconsider pipe/event/scenario targets. Use a frozen binary
target per experiment if necessary. Ordinary SMOTE does not directly handle the existing
multi-label vector; separate binary experiments must be declared, not silently introduced.
No invented abnormal-consumption class. Unknown labels are excluded/masked, not negatives.

**Deliver:** leakage report, target contract and class/event support table.
**Tests:** prohibited features are absent; source identities do not cross folds; simultaneous
events are represented correctly; a majority-class baseline exposes trivial targets.

## 4. Lock holdout data and construct ten chronological folds

For current BattLeDIM CSVs, default to Jan–Sep 2018 training/development, Oct–Dec 2018
held-out validation, 2019 locked test. For another paper dataset, freeze equivalent
date/scenario boundaries. Train/test-only designs still need internal validation.

Within the training/development portion, use ten expanding-window chronological splits.
Each fold trains on earlier data and assesses later data. Reserve a trailing slice of
fold-training for early stopping; the outer assessment block is not an epoch-selection set.
Reinitialize each model, transform, adjacency and sampler. All five models share fold IDs.

Purge using complete input/target intervals and event IDs, not only row indices.
For strict independent-window CV, the last time used by training windows must precede
the first time used by assessment windows. Also exclude or separately report shared
continuing leak events. Operational rolling evaluation may use already-observed history
across a boundary, but must be labeled separately. TimeSeriesSplit alone is not event purging.

For half-open [start, end), all included forecast target timestamps must be < end.
Document whether target_end means last included time or exclusive end. Group complete
scenarios and duplicated source records before windowing; never randomly shuffle windows.

**Deliver:** ten fold manifests with dates, purges, class counts, distinct events and
SMOTE-neighbor support. Keep operational/strict protocol labels separate.
**Tests:** exact-boundary and future-perturbation fixtures pass; no test data enter fitting.
If ten viable folds cannot be made, report every failed fold and reason. Seek enough
independent public scenarios or explicitly revise fold count; never silently drop folds
and report ten-fold performance.

## 5. Apply SMOTE inside fold-training only

Fit imputation/scaling/feature selection on real fold-training data, transform it, then
apply SMOTE. Transform all holdouts without resampling. Run no-SMOTE controls on identical
splits. Record k_neighbors, class counts and seed; minority count must exceed k_neighbors.
Absent classes cannot be manufactured to repair an infeasible fold.

For sequence models, declare the representation. A bounded common comparison may flatten
whole equal-length continuous windows, interpolate with SMOTE, then restore tensor shape.
Maintain channel/time order; exclude categorical IDs and unknown-label samples. Audit
ranges, temporal smoothing artifacts and source-event mixing. This is synthetic window
interpolation, not a physical simulation. If unsuitable, report the limitation and test
weighting/real-window resampling separately instead of pretending ordinary SMOTE is valid.

Fit M3's encoder within the fold on original training data. Its normal-only paper
reproduction remains separate from our supervised classification experiment. Train demand
forecasting on real windows, or mask regression loss for SMOTE classification samples;
do not invent future regression labels.

**Deliver:** fold-local sampler and synthetic provenance/count report.
**Tests:** held-out hashes/counts are unchanged; no held-out samples reach fit_resample;
neighbor failure is explicit; shapes/finite values are valid; no categorical or continuous
demand labels are accidentally interpolated.

## 6. Implement all five under one training interface

| ID | Required computation | Model-specific check |
| --- | --- | --- |
| M1 | Two Conv1D layers → unidirectional LSTM → head | Temporal/channel ordering |
| M2 | CNN → historical BiLSTM → attention → head | No post-origin samples in either direction; valid masks |
| M3 | Autoencoder training → latent sequence → LSTM → head | Fold-local encoder; both stages reload |
| M4 | Position encoding → Transformer → CNN → head | Sequence order/padding respected; no future tokens |
| M5 | Sensor adjacency → graph convolution → LSTM → head | Adjacency used; consistent node permutation preserves output |

M5 uses a small sensor graph estimated from real fold-training correlations, or fixed
metadata. Freeze a documented top-k/self-loop/normalization rule and sensor order.
Save adjacency as CSV. No graph UI, full hydraulic state reconstruction or pipe-localization
subsystem. Do not claim this reproduces a physical-network GNN paper.

Use the same features, targets, windows, folds and comparable tuning budgets. Different
windows/features belong in separate ablations. Optional ridge/gradient boosting/Isolation
Forest baselines do not replace any required hybrid. Demand regression may use separate
models or masked dual heads; evaluate it separately from classification.

**Deliver:** five modules, common trainer, configs and checkpointed smoke runs.
**Tests:** forward/backward shapes, finite gradients/loss, label masks, tiny-batch overfit,
checkpoint equivalence and CPU inference. Perturbing data after an origin cannot change
its prediction. Run bounded smoke tests before full training.

## 7. Diagnose, correct and retrain

Plot train/validation losses and task metrics by epoch. Also score original real training
examples, since balanced-training accuracy and natural-prevalence validation differ.

| Evidence | Candidate correction | Evidence to record |
| --- | --- | --- |
| Training improves while validation worsens | Earlier stopping, dropout, weight decay, smaller capacity | Gap and validation effect |
| Both errors stay high | Check labels/scaling/optimization, then capacity/context | Root cause and controlled change |
| Implausibly perfect results | Inspect leakage, duplicates, constant labels | Audit before acceptance |
| Large variation between folds | Inspect support, drift and scenario differences | Per-fold failures and counts |

Use the same CV protocol and bounded search budget to compare corrections. Reinitialize,
then choose final settings using development evidence. Refit on the declared final
training partition using a frozen epoch/early-stopping policy. At least three seeds
for final neural comparisons; save runtime and exact environment versions.

**Deliver:** initial/corrected comparison, learning curves, correction log and run recipe.
**Tests:** identical scoring rows, documented change reasons, no test-driven tuning.
A model losing to a simpler baseline is a valid result.

## 8. Calibrate and freeze alert rules

Calibrate on held-out real predictions at natural prevalence. Use disjoint calibration
and later method/threshold-selection subsets, or chronological out-of-fold predictions.
Do not automatically reuse November/December area labels: they are single-class.
Unsupported calibration stays unsupported; an uncalibrated score is not a risk percentage.

Specify persistence, gap merging, cooldown and one-to-one target-aware event matching.
The actionable alert begins at its confirming sample; merging cannot backdate it.
Declare valid-day/target-day denominator and handling of missing/censored events.

**Deliver:** calibrators, reliability bins, threshold/persistence settings and metric fixtures.
**Tests:** hand-calculated Brier/confusion/event scores; empty/no-event/single-class cases;
no calibration feedback to base-model fitting. Three five-minute consecutive exceedances
first confirm ten minutes after the initial exceedance.

## 9. Final evaluation and comparison

Freeze all models/transforms/thresholds before test outcomes are inspected. Evaluate
all five under identical eligibility rules. Never use final-test results for another
correction/retraining cycle without obtaining a new independent test for confirmation.

Publish four separate tables:
1. Paper-reported versus faithfully reproduced results, where equivalence is verified.
2. Common CV: per-fold and mean/std with valid/failed counts and class/event support.
3. Locked test: all hybrids, predeclared initial/corrected checkpoints and no-SMOTE controls.
4. Demand forecasting, if run, with identical observed-meter target/horizon.

Required classification fields: dataset/version, feature/target, model, SMOTE mode,
split/seed, accuracy, balanced accuracy, precision/recall, macro/per-class F1, support,
confusion matrix, average precision, ROC-AUC where defined. Add Brier, event recall,
false alerts/day, median/p90 delay among detected events, misses and runtime. Multi-label
accuracy must specify exact-match versus per-target scoring. Demand uses MAE/RMSE/bias.
Mark unsupported single-class metrics explicitly; accuracy alone is insufficient.

**Deliver:** saved predictions/configs, compact committed reports and exact run commands.
**Tests:** every metric has provenance and the same denominator across comparable models;
paper results are not substituted for our results; large datasets/checkpoints stay ignored.

## Transfer prompt

> Follow docs/HANDOFF_README.md for the revised five-hybrid CSV comparison. First resolve
> docs/PLAN_CHANGE_REVIEW.md prerequisites and freeze the paper/data/feature contract.
> Validate targets and ten chronological folds before fold-local SMOTE and training.
> No dashboard or full-network graph application. Distinguish exact reproduction from
> adaptations and report evidence for each acceptance condition.
