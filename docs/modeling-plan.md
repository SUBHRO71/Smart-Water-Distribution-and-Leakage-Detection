# Hybrid modeling and evaluation plan

## Decision

**Start with 1D-CNN + LSTM; establish its value empirically before adding complexity.** This is an engineering recommendation based on the data and project scope. No architecture can be called the best-performing model before controlled experiments.

| Hybrid from the problem statement | Role in this project | Tradeoff |
| --- | --- | --- |
| 1D-CNN + LSTM | First implementation: local hydraulic patterns plus temporal context | Practical starting scope; ignores explicit graph structure |
| CNN + BiLSTM + Attention | Ablation after the first model | More parameters; a BiLSTM is valid over an entirely historical window, but must never see post-origin data |
| Autoencoder + LSTM | Comparison when reliable leak labels are scarce | Reconstruction error detects novelty, not necessarily leaks; needs suitably clean training periods |
| Transformer + CNN | Later long-context experiment | Added training/tuning cost requires measurable benefit |
| GNN + LSTM | Topology-aware detection and localization extension | Requires accurate sensor-to-node/link mapping and treatment of unobserved nodes |

The problem's five techniques are candidates, not a substantiated universal ranking. The [BattLeDIM task](https://zenodo.org/records/3902046) supports both statistical and hydraulic approaches. Research on [water demand CNN/attention/LSTM modules](https://pmc.ncbi.nlm.nih.gov/articles/PMC11605409/) provides precedent for testing temporal hybrids, not proof they will win our leak task.

## Targets and observability

- **Demand forecast:** initially predict next-hour AMR consumption for observed meters or their covered-area sum. The 82 meters do not imply complete network consumption. Evaluate 24-hour forecasts later, potentially at hourly aggregation. Keep source-flow and customer-demand targets distinct.
- **Current leak:** detect active leaks at prediction time from trailing sensor history. Initially use the binary network label available from leak flow time series. Retain pipe-level labels separately for later localization. Audit the fraction of time with any leak: a persistently positive network label may be unsuitable for useful classification, requiring area/pipe targets instead.
- **Abnormal consumption:** no independent label in the primary dataset. Obtain utility event labels or create clearly identified hydraulic demand-change scenarios without leaks. Split by scenario before windowing. Do not rename all model residuals as abnormal consumption or present synthetic three-class results as field validation.
- **Risk score:** a calibrated probability of current leak when that is the trained label. Until calibration, expose only an anomaly score. A future-leak risk score needs its own horizon-specific labels and evaluation.
- **Early warning:** measure detection delay from onset for existing leaks. For future-event prediction, explicitly label whether an event starts in `(t, t+H]` and use observations only through `t`. Do not claim physical failure prediction from a detector trained on active leaks.

## First architecture (planned; not implemented)

1. Input a trailing 24-hour window (288 five-minute steps). Compare 6/24/48-hour contexts on validation only. Features: flow, pressure, AMR consumption, level, calendar values, causal changes and missingness masks. Fit scalers on training only.
2. Apply two temporal Conv1D layers (32 then 64 channels, kernel sizes 5 and 3), ReLU and dropout 0.1. Use causal padding for streaming intermediate outputs; all inputs must precede the prediction origin.
3. Feed sequence features to a unidirectional LSTM with 64 hidden units. Use its last hidden state; test temporal attention as a separate ablation.
4. A regression head produces the next 12 five-minute demand values. A separate sigmoid/logit head predicts current leak presence. Compare shared multi-task learning against separately trained heads to detect negative transfer.
5. Combine normalized Huber demand loss with weighted binary cross-entropy. Estimate class weights on training only, compare focal loss if necessary, and tune task weights using validation results.

Start with Adam, learning rate 0.001, batch size 64, maximum 50 epochs, validation early stopping and gradient clipping. These are starting settings, not optimized parameters. Record seeds, exact dependency versions, preprocessing configuration, source checksums and model checkpoints for every experiment. Use at least three seeds for neural comparisons.

## Chronological experiment design

Use January–September 2018 for fitting and October–December 2018 for validation, leaving all of 2019 for a final locked evaluation. Select scalers, imputation rules, class weights, architectures, calibration and alert thresholds without 2019 data. Within the validation period, reserve separate earlier calibration and later threshold-selection blocks, or use chronological out-of-fold predictions.

Assign windows by their prediction target times. Purge training windows whose future targets enter validation; never randomly split overlapping windows. Historical context before the boundary is allowed at validation/test origins because it would already have been observed. For stricter independent-event assessment, report events spanning boundaries separately, and also evaluate held-out complete events/scenarios. Persistent unresolved leaks can carry across years and reduce event independence.

For the eventual multi-step model, explicitly distinguish rolling-origin from fixed-origin forecasts. Never consume future observed demand; BWDF's perfect future weather protocol must be reported separately from realistic weather forecasts. Use past-only forward fill with a stated maximum gap, training medians and masks; no backward fill or interpolation across future samples. The starter deliberately rejects gaps until that policy is implemented.

Ground-truth leak CSVs, future repairs, simulation leak schedules, `L-TOWN_Real.inp` and full-year normalization statistics are prohibited predictors. Full 2018 ground-truth supervision must be disclosed as a retrospective experiment that has more information than the original challenge setting.

## Baselines and metrics

1. **Implemented:** daily/weekly seasonal persistence for rolling one-step flow forecasts, reporting MAE and RMSE in m3/h.
2. **Next forecasting baselines:** persistence, calendar/lag ridge regression and gradient boosting; compare to standalone LSTM and CNN + LSTM under identical windows and forecast horizons. Include per-sensor and aggregate MAE/RMSE and forecast bias; avoid MAPE when demand is near zero.
3. **Next detection baselines:** validation-tuned residual thresholds, pressure/flow change rules, Isolation Forest and an autoencoder. Ensure presumed normal training periods are not silently contaminated with known leak events.
4. **Leak metrics:** PR-AUC, precision, recall and F1 at a fixed validation-selected threshold, false alarm events/day, event recall and median/90th-percentile onset-to-alert delay. Report leak prevalence and missed events; accuracy alone is insufficient.
5. **Alert protocol:** tune persistence (initial candidate: three consecutive samples), merge near-adjacent alerts under a fixed gap rule and match alerts to events consistently. For network labels merge overlapping leak intervals; use pipe-matched events only when localization exists. Report the same alert rules for all models.
6. **Risk calibration:** fit sigmoid or isotonic calibration using held-out predictions, then evaluate Brier score and reliability bins on the final test. A percentage is not trustworthy merely because the output passes through a sigmoid.

Choose the hybrid only if improvements over the strongest simple baseline are consistent across seeds and meaningful under the false-alarm budget. Bootstrap confidence intervals by complete days/events, not individual correlated samples. Report performance by leak size, abrupt versus gradual onset, and sensor availability where metadata permit.

## Delivery sequence and completion criteria

| Phase | Deliverable | Completion evidence |
| --- | --- | --- |
| 0: starter | Downloader, preprocessing, baseline, source-backed plan | CLI runs on original data; unit and time-integrity checks pass |
| 1: audit | EDA, incident inventory, clean split specification | Missingness, units, label balance and event overlap documented |
| 2: models | Conventional baselines and CNN + LSTM training pipeline | Reproducible validation table and saved configurations |
| 3: detection | Thresholded alerts and calibrated risk | Event-based held-out metrics, false-alarm budget and calibration plot |
| 4: extensions | Attention/GNN ablations and abnormal-demand scenarios | Controlled comparisons, distinct real/synthetic provenance |
| 5: app | Forecast/alert dashboard with sensor evidence | End-to-end replay without future-data access |

Pumps and valves should not be controlled by this research classifier. A distribution optimizer would be a separate phase with hydraulic simulation, pressure limits, tank reserves, pump/energy constraints and operator validation.
