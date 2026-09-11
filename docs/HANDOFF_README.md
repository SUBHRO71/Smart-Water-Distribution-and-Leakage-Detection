# Implementation handoff: remaining research work

This is an execution guide for the next coding model. It defines work and acceptance
conditions; it does not mean the work has been implemented or the conditions have passed.

There are **four phases**: audit, models, detection/calibration, and extensions.
Phases 1-3 form the core study. Phase 4 contains separately evaluated extensions.
Complete the steps in order; do not start long training runs before the data and
split checks pass. A failed model comparison is a valid result, not a reason to hide it.

## Start here

1. Read the root [README](../README.md), [modeling plan](modeling-plan.md),
   [dataset notes](datasets.md), and [existing verification](verification.md).
2. Inspect the actual repository and any applicable `AGENTS.md` instructions. Preserve
   existing work. Treat the artifact paths below as proposed paths, not existing commands.
3. Confirm the available data and environment. Reuse the existing downloader, loader,
   preparation and seasonal baseline instead of rebuilding them.
4. Run the existing offline checks before implementation:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

The starter has a verified full 2018 download, 105,120 time rows, 119 sensor channels,
three calendar features, and separate aggregate labels. These are recorded observations,
not a completed EDA. Confirm local availability; downloads and reports are ignored by Git.
2019 remains the test set. No hybrid model or calibrated detector exists yet.

## Rules shared by every phase

- Use BattLeDIM as the primary dataset. Do not join unrelated BWDF weather to L-Town.
- Preserve original data, units and source checksums. AMR demand is converted from
  L/h to m3/h; pressure is measured as head in meters, not automatically bar.
- Use sensor observations available at the forecast origin. Ground-truth leak flows,
  repairs, simulation schedules and the real network file are never input features.
- Fit preprocessing, feature selection and class weights on training data only.
  Future interpolation, backward fill and random splits of overlapping windows are prohibited.
- Separate implemented code, smoke-test results and full experimental results.
  Do not invent utility labels, model scores, area mappings or event coverage percentages.
- Save configurations, seeds, exact environment versions, split manifests, metrics and
  checkpoints. Commit compact summaries under `docs/results/`; keep raw data, large
  predictions and checkpoints in ignored folders. Do not force-add them to Git.
- Every phase ends with a short status record: changed files, commands, acceptance
  results, limitations, and the exact next step. Mark an item blocked if its required
  data are unavailable, while continuing independent work.

## Phase 1: EDA and audit

### Step 1. Audit raw sensor coverage before cleaning

Produce a per-channel table with type, node/link ID, unit, first/last observation,
expected and observed timestamps, missing cells, missing intervals, duplicates,
nonfinite values, longest gap, constant/stuck runs and suspicious ranges. Plot coverage
and representative daily/weekly profiles. Map sensors to the nominal network and
distinguish temporal completeness from spatial coverage.

The existing strict loader rejects gaps; add an audit path that can report malformed
data before rejection. Do not silently repair the source or interpret an absent
measurement as zero. Separate anomalies requiring review from confirmed sensor faults.

**Deliver:** `reports/audit/channel_coverage.csv`, a coverage figure and a compact
`docs/results/phase1-audit.md` summary.

**Acceptance:** a tiny fixture with a missing timestamp, a missing value, a duplicate
and a stuck channel reports each correctly. Counts reconcile against the expected
five-minute grid. Every channel has a mapped ID or an explicit unresolved reason.

### Step 2. Build a pipe-level leak event inventory

Extract contiguous positive leak-flow intervals per pipe using a documented activity
threshold. Start with greater than zero for the published series; justify any tolerance.
Treat intervals as `[onset, end_exclusive)`. At five-minute cadence a one-sample event
lasts five minutes. Missing label samples are unknown, not evidence that a leak stopped.

Record event ID, pipe ID, endpoints, area where supported, onset, last active sample,
end exclusive, duration, peak/mean leak flow, integrated volume, label source and
left/right censoring. Volume is the sum of m3/h times 5/60 hours over observed intervals;
flag incomplete volumes. Keep reported repair time separate from inferred onset.
Do not infer hole diameter from flow. Preserve overlapping events on different pipes.

**Deliver:** `reports/audit/leak_events.csv` and mapping/unresolved-ID tables.

**Acceptance:** fixtures cover one-sample, repeated, overlapping, missing-label and
boundary-censored events. Event masks reconstruct the known positive label samples.
All pipe IDs map to nominal links or appear in an unresolved table. Duration and volume
calculations pass hand-computable examples.

### Step 3. Measure event label coverage and design useful targets

Report two separate quantities: timestamp prevalence and event-level label coverage.
For event coverage, declare a reference inventory and count matched labeled events
divided by eligible reference events. Report by onset month, area and split; state how
censored and partially labeled events enter the denominator. Compare original repair
reports with retrospective full labels if both are available, using explicit matching rules.

An inventory extracted from leak labels cannot independently establish that all actual
events were labeled. If no independent reference exists, report completeness as
**unknown**, plus the observed event count and label provenance. Do not report 100%
by comparing a label-derived inventory to itself.

The network-wide label is positive 97.8% of the time. Compare area and pipe targets
using training/development data: positive/negative duration, distinct event count,
normal coverage and sensor observability. Prefer defensible area targets if individual
pipes lack observability. Freeze area definitions from network evidence; document
cross-area pipes rather than guessing. A nearby sensor does not guarantee observability.

Specify a `K`-target binary vector for simultaneous leaks, not a mutually exclusive
softmax. Preserve unknown-label masks and exclude unknowns from loss and scoring.
Keep targets with no development events identified as unsupported rather than claiming
their probabilities are calibrated. Define which AMR meters form the demand target;
their sum is observed-meter consumption, not necessarily whole-network demand.

**Deliver:** `reports/audit/label_balance.csv`, `configs/targets.json` and a target
decision section in the audit summary.

**Acceptance:** event and timestamp denominators are explicit; simultaneous leaks
produce multiple positives; unknowns stay unknown. Targets and exclusions are reproducible,
and the nearly constant aggregate label is not the main classifier target.

### Step 4. Freeze chronological splits and purging

Use these half-open source-time intervals, preserving the source's unspecified timezone:

| Partition | Interval | Permitted use |
| --- | --- | --- |
| Training | `[2018-01-01, 2018-10-01)` | Model/preprocessing fitting |
| Validation: model selection | `[2018-10-01, 2018-11-01)` | Hyperparameters, early stopping, ablations |
| Validation: calibration | `[2018-11-01, 2018-12-01)` | Fit calibration functions |
| Validation: alert selection | `[2018-12-01, 2019-01-01)` | Calibration-method and alert-threshold selection |
| Final test | `[2019-01-01, 2020-01-01)` | Locked final evaluation only |

If monthly blocks lack required classes/events, define chronological out-of-fold
alternatives using 2018 only and record the change before training. Never substitute
2019 for a deficient validation block.

For every sample store input start/end, origin, forecast target start/end and event IDs.
For origin `t`, history ends at `t`; demand targets are `t+5 min` through `t+60 min`.
All forecast target samples must stay inside their assigned partition. Remove training
windows with targets crossing a boundary. Past context before a boundary is allowed
in the operational rolling evaluation; future context is never allowed.

Also define a strict event-independent development comparison: purge any earlier-block
window whose input or targets touch a leak event continuing into the next development
block. Use interval intersection over the entire event, not only a fixed one-hour gap.
Apply the rule at model/calibration/threshold boundaries and publish excluded event/window
counts. An event continuing at the end of 2018 is censored and cannot be treated as a
fresh independent event in 2019. Use development evidence for fitting exclusions; do not
inspect 2019 labels to change the trained model. At final evaluation report carryover
events separately from newly starting test events under the predeclared rule.

**Deliver:** `configs/splits.json`, a sample split manifest, purge counts and a written
distinction between operational and strict event-independent results.

**Acceptance:** boundary fixtures prove that no future target crosses partitions,
development events do not contaminate the strict comparison, and known historical
context remains allowed in the operational comparison. Changing held-out values cannot
change fitted training preprocessing. If purging removes most usable events, document
the limitation instead of silently weakening the rule.

**Phase 1 gate:** coverage, event inventory, target contract and split/purge tests pass
before model training begins. Do not perform exploratory model selection on 2019.

## Phase 2: conventional and hybrid models

### Step 5. Establish comparable conventional baselines

Implement ridge regression and gradient boosting for the same demand targets and
forecast horizons. Use causal lags, rolling summaries and calendar features. Recompute
seasonal persistence on those targets; the existing p227 flow score is not directly
comparable to AMR demand forecasts.

Implement Isolation Forest on training-only hydraulic features. Document whether it
fits presumed-normal area periods or contaminated data, and use held-out data to choose
alert thresholds. Its anomaly score is not a leak probability. Add a simple residual
threshold detector as a reference. Retain prediction origins and target IDs in outputs.

**Deliver:** reusable baseline modules, configurations and a validation metric table.

**Acceptance:** synthetic periodic data reproduce seasonal predictions; all predictors
use past-only features; models reload with identical predictions. MAE/RMSE are computed
in physical units on identical forecast rows. Detection-score direction is explicit.

### Step 6. Implement CNN + unidirectional LSTM

Start with 288 historical steps, two Conv1D layers (32/64 channels, kernels 5/3),
ReLU, dropout 0.1 and a unidirectional LSTM with 64 hidden units. Use two heads:
next 12 demand steps and `K` current-leak logits from the Phase 1 target contract.
Use normalized Huber regression loss plus training-weighted binary cross-entropy
with logits, applying unknown-label masks. Tune task weights on model-selection data.

Build a configurable training/evaluation pipeline with chronological window loading,
training-only transforms, seeds, early stopping, checkpoint save/load and inference.
Run a small smoke test first, then bounded experiments. Compare single-task heads to
the dual-head model. Record runtime and at least three seeds for final neural comparisons.

**Deliver:** model module, training entry point, saved configurations, checkpoints and
2018 validation predictions. Document exact runnable commands when they exist.

**Acceptance:** verify tensor shapes, finite losses/gradients, mask behavior, a tiny-batch
overfit check, checkpoint equivalence and CPU inference. Perturbing observations after
an origin must not alter that origin's prediction. Training logs confirm that early
stopping uses model-selection data only. Report weak results honestly; beating a baseline
is an experimental finding, not a condition for claiming the code is implemented.

## Phase 3: detection, calibration and evaluation

### Step 7. Define alerts and event matching

Use per-target thresholds, a persistence count (initial candidate: three consecutive
samples), and a documented short-gap merge rule. The first actionable alert time is
the confirming sample, not the first sample in the persistence run. Missing readings
break persistence unless another causal rule is explicitly justified.

Specify alert end behavior, cooldown, gap tolerance and target matching before scoring.
Merged reporting intervals must retain the actual issuance times; merging must not
backdate detection. Match alerts to events one-to-one within the declared temporal
overlap/tolerance and the same area/pipe target. Define duplicate alerts and censored
events explicitly. One alert must not count as detection of several independent events.

**Deliver:** alert/event tables, matching logic and configuration.

**Acceptance:** hand-built fixtures cover isolated spikes, sustained alarms, missing
samples, short/long gaps, overlapping target events, duplicate alerts, missed events
and boundary events. Three-sample persistence produces the expected 10-minute delay
when a signal first exceeds its threshold at onset on a five-minute grid.

### Step 8. Calibrate and report risk

Fit sigmoid and isotonic candidates using calibration-block predictions from a frozen
model. Choose the method and operating threshold using the later selection block with
a stated false-alarm budget and event-recall tradeoff. Define the budget in configuration
before selecting a winner; do not invent an achieved performance requirement.

Record uncalibrated and calibrated Brier scores, reliability bins with sample counts,
PR curves, precision/recall and event metrics. State whether "PR-AUC" means average
precision or trapezoidal area; use one definition consistently. Report per-target values
and macro/micro aggregates with their class support. Mark single-class PR comparisons
and unsupported per-target calibration as undefined rather than manufacturing scores.

False alarms/day = unmatched issued alert events divided by monitored valid days under
the published matching policy. State whether the denominator is calendar days or
target-days. Report event recall, median/p90 delay for detected events, missed-event
counts and censored-event handling. Reliability plots must show bin populations.

**Deliver:** calibration artifacts, reliability tables/plot and validation evaluation report.

**Acceptance:** calibrated outputs lie in `[0,1]`; test fixtures verify Brier score and
event metrics by hand. Empty alerts, no events, one-class data and empty reliability bins
have explicit behavior. No calibration fitting or threshold selection consumes test data.

**Phase 3 gate:** the core pipeline is ready for a locked evaluation. If Phase 4 model
comparisons are planned, complete their 2018 selection first, then evaluate all frozen
candidates on 2019 together. Do not repeatedly consult 2019 to decide the next model.

## Phase 4: separately evaluated extensions

### Step 9. CNN + BiLSTM + Attention ablation

Reuse the core input/target/split contracts. Compare the base model, BiLSTM alone,
attention alone and their combination where the compute budget permits. Bidirectionality
is confined to the historical window. Hold tuning budgets and reporting protocols comparable.

**Acceptance:** the future-perturbation test still passes. Report seed variation,
parameter count/runtime and forecasting/detection tradeoffs against the base model.

### Step 10. Autoencoder + LSTM comparison

Define the hybrid concretely: encode each historical sensor vector with an autoencoder,
then feed the latent sequence to an LSTM forecaster. Compare reconstruction and forecast
residuals as anomaly scores. Fit on verified normal training-area intervals where available;
if those are insufficient, document contamination or defer the normal-only experiment.

**Acceptance:** training excludes held-out samples; reconstruction error is not described
as calibrated risk without Step 8. Controlled normal/anomalous fixtures exercise scoring,
and results use the same alert/matching protocol as the core model.

### Step 11. GNN + LSTM topology-aware variant

Use the nominal graph: junctions as nodes and pipes as edges, with observed-sensor masks
and explicit link-flow mapping. Specify whether the head aggregates node embeddings to
areas or predicts pipe labels. Never fill missing sensors with ground-truth hydraulic
states. Use graph embeddings over time as input to the LSTM.

**Acceptance:** IDs/edges are validated, graph batching and dimensions pass tests,
and consistently permuting node order leaves predictions correspondingly unchanged.
Compare with the temporal model under matched information; flag mapping/calibration
uncertainty. Better localization must be demonstrated, not assumed from using a graph.

### Step 12. Abnormal consumption scenarios and utility labels

Generate documented demand-change scenarios with hydraulic simulation so pressure,
flow and consumption remain consistent. Vary demand while keeping leak status controlled;
include both demand-only and coincident demand/leak cases. Split by complete scenario
and simulation seed before making windows. Report synthetic and utility results separately.

Utility labels require an actual supplied dataset, permission and an event schema.
If absent, leave utility validation explicitly blocked; synthetic labels cannot satisfy it.
Define how Normal / Leak / Abnormal consumption handles coincident events (prefer
multi-label outputs or a declared combined category), rather than forcing false exclusivity.

**Acceptance:** provenance, units, hydraulic validity and scenario separation are tested.
No source scenario crosses splits. Demand-only fixtures do not acquire leak labels by
construction, and a classifier's success is measured on held-out scenarios.

## Final evaluation and transfer checklist

Freeze selected architectures, transforms, calibration and alert settings before the
2019 evaluation. Compare all frozen candidates on the same eligible rows/events;
report carried-over and newly starting events separately, along with purge/exclusion
counts. Include simple baselines, uncertainty across seeds or day/event resampling,
and limitations. Never promise a target accuracy before observing results.

- [ ] Phase 1: coverage, event inventory, target definition and split tests complete.
- [ ] Phase 2: conventional models and core hybrid reproducible; validation results saved.
- [ ] Phase 3: causal alerts, calibration and metric fixtures pass.
- [ ] Phase 4: each extension marked completed, deferred or blocked with evidence.
- [ ] Locked test results and exact reproduction commands documented.
- [ ] Existing tests/lint pass; compact findings committed; large artifacts kept separate.

Suggested prompt to transfer this work:

> Read docs/HANDOFF_README.md and the linked project documents. Implement Phase 1
> first, following its acceptance conditions. Reuse the starter and preserve existing
> changes. Do not train models until the audit, target design and split checks pass.
> Record deliverables, verification evidence and unresolved data dependencies before
> moving to the next phase. Treat later phase checklists as planned work, not completed work.
