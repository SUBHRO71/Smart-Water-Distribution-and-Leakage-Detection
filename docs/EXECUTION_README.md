# Research Execution Log: Smart Water Distribution and Leakage Detection

> **Protocol update, 2026-09-11:** The Phase 1 BattLeDIM work below is retained as an
> audit trail. Its area-label feasibility gate was later found to fail, so “Gate 1
> Passed” statements in the historical entries do not authorize classification training
> on those targets. The five-model classification track now uses LeakDB Hanoi_CMH. Its
> acquisition, audit, frozen windows, grouped folds, fold-local scaling and training-only
> SMOTE smoke test are complete. See the current [README](../README.md) and
> [feasibility report](results/prerequisite-and-feasibility.md).

This document records the step-by-step execution of the research handoff detailed in [HANDOFF_README.md](HANDOFF_README.md). For every step completed, it documents the steps taken, decisions and justifications made, deliverables produced, and verification test results.

---

## Execution Status Matrix

| Phase | Step | Description | Status | Completion Date / Evidence |
|---|---|---|---|---|
| **Phase 1** | **Step 1** | Audit raw sensor coverage before cleaning | Completed | 2026-09-11 / 2 tests passed, 119 channels mapped |
| | **Step 2** | Build a pipe-level leak event inventory | Completed | 2026-09-11 / 3 tests passed, 14 events extracted |
| | **Step 3** | Measure event label coverage and design useful targets | Completed | 2026-09-11 / 3 tests passed, targets.json frozen |
| | **Step 4** | Freeze chronological splits and purging | Completed | 2026-09-11 / 3 tests passed, splits.json frozen, Gate 1 Passed |
| **Phase 2** | **Step 5** | Establish comparable conventional baselines | Next / Ready | — |
| | **Step 6** | Implement CNN + unidirectional LSTM | Pending | — |
| **Phase 3** | **Step 7** | Define alerts and event matching | Pending | — |
| | **Step 8** | Calibrate and report risk | Pending | — |
| **Phase 4** | **Step 9** | CNN + BiLSTM + Attention ablation | Pending | — |
| | **Step 10** | Autoencoder + LSTM comparison | Pending | — |
| | **Step 11** | GNN + LSTM topology-aware variant | Pending | — |
| | **Step 12** | Abnormal consumption scenarios & utility labels | Pending | — |

---

## Environment & Initial Baseline Confirmation

- **Operating System**: Windows
- **Python**: 3.13.7 (in `.venv`)
- **Pytest**: 9.1.1 (configured with `--basetemp=pytest_tmp -p no:cacheprovider` to handle Windows temp directory permissions)
- **Ruff**: 0.16.7 (All checks passed)
- **Initial Verification**: 9/9 starter tests passing.
- **Dataset**: Full 2018 BattLeDIM release (105,120 rows, 119 sensor channels, 3 calendar features, ground-truth leak flow series).

---

## Phase 1: EDA and Audit Execution

### Step 1: Audit Raw Sensor Coverage Before Cleaning

- **Objective**: Audit temporal completeness, sensor integrity, and spatial mapping of raw SCADA streams against nominal network topology before applying strict cleaning or silent filling.
- **Steps Taken**:
  1. Implemented network topology parser (`parse_network_topology`) in `src/smart_water/audit.py` to extract junctions, pipes, pumps, valves, and reservoirs from `data/raw/battledim/L-TOWN.inp`, mapping them into hydraulic partitions (Area A, B, C).
  2. Implemented time series auditor (`audit_time_series`) to calculate expected vs. observed timestamps on a uniform 5-minute grid, missing cells, missing intervals, duplicates, nonfinite values, longest gaps, stuck runs, and physical range plausibility.
  3. Added `smart-water audit` CLI command in `src/smart_water/cli.py` to audit all 119 channels across the 4 SCADA files.
  4. Created visual asset generator (`generate_coverage_svg`) producing a standalone vector graphic for coverage profiling.
  5. Implemented unit tests in `tests/test_audit.py` covering synthetic failure modes and nominal dataset reconciliation.
- **Decisions & Justifications**:
  - **Network Topology Partitioning**: Derived hydraulically from `L-TOWN.inp` graph connectivity rather than arbitrary spatial heuristics:
    - *Area A*: Main distribution network (657 junctions, 29 pressures, 2 reservoir flows).
    - *Area B*: Pressure-reduced zone (31 junctions, 1 pressure sensor downstream of PRV-3: `n229 -> n226`).
    - *Area C*: Elevated zone (93 junctions, 3 pressures, tank T1, all 82 AMR meters, pumped via `PUMP_1: n54 -> T1`).
  - **No Silent Repairs**: The audit module reports malformed data without altering or discarding rows, providing full traceability before pipeline ingestion.
- **Verification Evidence & Test Results**:
  - `tests/test_audit.py` executed:
    - `test_audit_catches_missing_timestamp_value_duplicate_and_stuck_channel`: PASSED (correctly isolated 1 missing cell, 1 missing interval with 20-min gap, 1 duplicate timestamp, 1 stuck channel of 10 steps, and 1 negative pressure anomaly).
    - `test_topology_partitioning_and_channel_mapping`: PASSED (verified all 119 channels mapped with 0 unresolved, exactly 105,120 rows per channel, 0 missing cells, 0 duplicates, 0 missing intervals).
  - Pytest result: `2 passed in 7.53s` (plus 9 initial tests: `11 passed`).
- **Deliverables Produced**:
  - Per-channel audit table: [`reports/audit/channel_coverage.csv`](../reports/audit/channel_coverage.csv)
  - Coverage figure: [`reports/audit/sensor_coverage_profile.svg`](../reports/audit/sensor_coverage_profile.svg)
  - Audit report: [`docs/results/phase1-audit.md`](results/phase1-audit.md)
- **Gate Acceptance**: Meets all Step 1 acceptance criteria. All 119 channels reconcile against the expected 5-minute grid with verified mappings.

### Step 2: Build a Pipe-Level Leak Event Inventory

- **Objective**: Extract contiguous positive leak-flow intervals per pipe using an explicit activity threshold ($>0.0\ \text{m}^3/\text{h}$), map endpoints to nominal links, integrate volume physically, and track boundary censoring.
- **Steps Taken**:
  1. Implemented event extraction engine (`extract_leak_events`) in `src/smart_water/inventory.py` to identify contiguous runs $[t_{\text{onset}}, t_{\text{end\_exclusive}})$, tracking duration, peak/mean leak flow, integrated volume ($\sum \text{flow} \times \frac{5}{60}\ \text{h}$), and left/right boundary censoring.
  2. Implemented label mask reconstructor (`reconstruct_positive_mask`) to verify that the extracted events losslessly reproduce positive ground-truth label samples.
  3. Added `smart-water inventory` CLI command in `src/smart_water/cli.py`.
  4. Created comprehensive unit test suite in `tests/test_inventory.py` covering hand-computable fixtures, repeated/overlapping events, censoring, and full 2018 dataset reconstruction.
- **Decisions & Justifications**:
  - **Interval Definition**: Strictly formulated as half-open $[t_{\text{onset}}, t_{\text{end\_exclusive}})$. At a 5-minute cadence, an isolated 1-sample event active at $00:05$ has an onset of $00:05$, end exclusive of $00:10$, and duration of exactly $5.0$ minutes ($0.0833\ \text{h}$).
  - **Volume Integration**: Evaluated in physical units ($\text{m}^3$) by trapezoidal/rectangular step integration: $\text{volume} = \sum (\text{flow in m}^3/\text{h}) \times (5/60\ \text{h})$.
  - **Boundary Censoring**: Explicitly tracks 4 long-running leaks that are right-censored at the end of 2018 (`p257` in Area C; `p427`, `p654`, `p810` in Area A). In accordance with the handoff protocol, these cannot be treated as fresh independent events in 2019.
- **Verification Evidence & Test Results**:
  - `tests/test_inventory.py` executed:
    - `test_inventory_hand_computable_fixtures`: PASSED (verified exact volume calculations: $1.0\ \text{m}^3$, $2.0\ \text{m}^3$, $0.5\ \text{m}^3$, $3.0\ \text{m}^3$; left/right censoring; and multi-sample overlaps).
    - `test_unresolved_pipe_handling`: PASSED (verified unmapped pipe IDs are safely routed to unresolved table).
    - `test_full_2018_leak_event_inventory_and_mask_reconstruction`: PASSED (extracted exactly 14 events across 14 pipes, 0 unresolved pipes, and reconstructed the ground-truth positive mask with 100.0% exact equality across all 105,120 rows).
  - Pytest result: `3 passed in 0.75s` (cumulative: `14 passed`).
- **Deliverables Produced**:
  - Pipe-level leak inventory: [`reports/audit/leak_events.csv`](../reports/audit/leak_events.csv)
  - Unresolved pipe mapping table: [`reports/audit/unresolved_pipe_ids.csv`](../reports/audit/unresolved_pipe_ids.csv) (0 unresolved pipes).
- **Gate Acceptance**: Meets all Step 2 acceptance criteria. Event masks reconstruct the known positive labels perfectly; volume and duration match hand-calculated fixtures; all pipes map to nominal links.

### Step 3: Measure Event Label Coverage and Design Useful Targets

- **Objective**: Quantify timestamp prevalence vs. event-level coverage across partitions and areas, reject uninformative aggregate network labels, and formalize a multi-label target contract with simultaneous leak support and causal AMR demand forecasting.
- **Steps Taken**:
  1. Implemented label balance calculator (`compute_label_balance`) in `src/smart_water/targets.py` to calculate positive/negative timestamp prevalence, active event counts, and new onset counts for each partition and hydraulic area.
  2. Defined target configuration contract generator (`build_targets_config`) freezing `configs/targets.json`.
  3. Implemented multi-label binary target matrix constructor (`create_target_matrix`) for $[y_{\text{Area\_A}}, y_{\text{Area\_B}}, y_{\text{Area\_C}}]$.
  4. Added `smart-water targets` CLI command in `src/smart_water/cli.py`.
  5. Implemented unit tests in `tests/test_targets.py` covering multi-label independence, simultaneous leak handling, and schema validation.
  6. Added Target Decision and Label Provenance section to [`docs/results/phase1-audit.md`](results/phase1-audit.md).
- **Decisions & Justifications**:
  - **Rejection of Aggregate Label**: Network-wide `is_leak` is positive on 97.8025% of all 2018 timestamps (and 100.0% of Oct, Nov, and Dec validation timestamps). A classifier predicting constant positive achieves 97.8% accuracy without learning any hydraulic pattern. It is strictly excluded as a modeling target.
  - **Multi-Label Binary Target Vector ($K=3$)**: Instead of single-label binary or mutually exclusive softmax, we define independent binary probabilities per area target: $\mathbf{y}_t \in \{0, 1\}^3$. Simultaneous leaks (e.g. Area A and Area C both leaking) naturally yield multi-positive outputs $[1, 0, 1]$.
  - **Demand Target Formulation**: Aggregated next-hour AMR consumption across all 82 customer meters in Area C: $\sum_{m=1}^{82} d_{m, t+5\text{m}:t+60\text{m}}$ (12 steps). It represents observed metered consumption, not total network demand.
  - **Coverage Completeness Provenance**: Reported explicitly as *Unknown* (with provenance noted) because BattLeDIM provides no independent field repair log to establish ground-truth unobserved leak absence, avoiding circular claims of 100% completeness.
- **Verification Evidence & Test Results**:
  - `tests/test_targets.py` executed:
    - `test_simultaneous_leaks_and_multilabel_independence`: PASSED (verified independent multi-positive output $[1, 0, 1]$ when Area A and C leak concurrently).
    - `test_targets_contract_specification`: PASSED (verified 82 AMR meters, 12 steps, Huber loss, BCEWithLogitsLoss, and rejection policy).
    - `test_label_balance_on_2018_splits`: PASSED (verified Area B is 0.0% positive in validation; Area A has active and onset events in validation; completeness is classified as Unknown).
  - Pytest result: `3 passed in 0.73s` (cumulative: `17 passed`).
- **Deliverables Produced**:
  - Label balance report: [`reports/audit/label_balance.csv`](../reports/audit/label_balance.csv)
  - Target contract configuration: [`configs/targets.json`](../configs/targets.json)
- **Gate Acceptance**: Meets all Step 3 acceptance criteria. Timestamp denominators are explicit; simultaneous leaks yield independent positives; unknowns remain masked; the near-constant aggregate label is rejected.

### Step 4: Freeze Chronological Splits and Purging

- **Objective**: Freeze chronological half-open partitions $[t_{\text{start}}, t_{\text{end}})$, enforce causal window boundaries (forecast target horizons cannot cross partition ends), implement strict event purging, and publish window eligibility manifests.
- **Steps Taken**:
  1. Implemented split interval schedules (`SPLIT_INTERVALS`) and split configuration builder (`build_split_configuration`) in `src/smart_water/splits.py`.
  2. Implemented rolling window manifest generator (`generate_window_manifest`) tracking 288-step (24h) history, 12-step (1h) forecast targets, boundary crossing flags, and event contamination checks.
  3. Added `smart-water splits` CLI command in `src/smart_water/cli.py`.
  4. Implemented unit tests in `tests/test_splits.py` verifying boundary target containment, strict event purging, and training preprocessing isolation from held-out validation data.
  5. Documented the operational vs. strict event-independent protocols and purge statistics in [`docs/results/phase1-audit.md`](results/phase1-audit.md#4-chronological-splits-boundary-purging--manifest).
- **Decisions & Justifications**:
  - **Half-Open Partitions**: Defined as $[t_{\text{start}}, t_{\text{end}})$ preserving the source's unspecified timezone:
    - *Train*: `[2018-01-01 00:00:00, 2018-10-01 00:00:00)` (78,624 timestamps)
    - *Val: Model Selection*: `[2018-10-01 00:00:00, 2018-11-01 00:00:00)` (8,928 timestamps)
    - *Val: Calibration*: `[2018-11-01 00:00:00, 2018-12-01 00:00:00)` (8,640 timestamps)
    - *Val: Alert Selection*: `[2018-12-01 00:00:00, 2019-01-01 00:00:00)` (8,928 timestamps)
    - *Final Test*: `[2019-01-01 00:00:00, 2020-01-01 00:00:00)` (Locked evaluation only)
  - **Operational Protocol**: For origin $t$, historical context preceding partition start is permitted (since it was physically observed), but forecast targets cannot cross partition ends. Exactly 11 target-crossing windows per partition boundary are truncated. Total eligible operational windows in 2018: **104,789**.
  - **Strict Event-Independent Purging & Limitation**: Purges any earlier-block window whose input history or forecast target overlaps an event that continues into the subsequent partition (`p257` in Area C; `p427`, `p654`, `p810` in Area A). Because `p257` is active from Jan 9 through year-end, strict purging leaves 2,012 windows in Train and 0 in Validation blocks. As required by the handoff rules, this limitation is explicitly published rather than weakening the rule.
- **Verification Evidence & Test Results**:
  - `tests/test_splits.py` executed:
    - `test_window_boundary_enforcement_and_past_context`: PASSED (boundary window at 23:00 has target end 00:00, eligible=True; window at 23:05 crosses boundary, eligible=False; Oct 1 origin allows late-Sept past history).
    - `test_strict_event_purging`: PASSED (window touching boundary-crossing event is operational=True, strict=False with purge reason recorded).
    - `test_preprocessing_isolation_from_held_out_data`: PASSED (extreme corruption of held-out validation values produces 0.0 change in fitted training parameters).
  - Pytest result: `3 passed in 0.58s` (cumulative: `20 passed`).
- **Deliverables Produced**:
  - Splits configuration: [`configs/splits.json`](../configs/splits.json)
  - Representative split manifest: [`reports/audit/split_manifest_sample.csv`](../reports/audit/split_manifest_sample.csv)
  - Audit report update: [`docs/results/phase1-audit.md`](results/phase1-audit.md#4-chronological-splits-boundary-purging--manifest)
- **Gate Acceptance (Phase 1 Gate)**: **PASSED**.
  - Sensor coverage audit verified (119 channels mapped, 0 unresolved, 105,120 rows).
  - Pipe-level leak inventory verified (14 events, hand calculations verified, mask reconstructed 100%).
  - Multi-label target contract frozen (`targets.json`, $K=3$, AMR sum, near-constant label rejected).
  - Chronological splits and purging rules frozen (`splits.json`, manifest sample verified, no future leakage).
  - Full pytest suite (20 tests) and Ruff checks passing cleanly.
