# Phase 1: EDA and Sensor Coverage Audit Summary

Historical snapshot from the first audit. The area-label gate below was subsequently
rejected for classification; the original split counts include a corrected endpoint bug.
Use the [current feasibility report](prerequisite-and-feasibility.md) and
[main README](../../README.md) for decisions. In the old demand wording below,
"cumulative consumption" denotes a sequence of aggregate AMR flow rates, not volume.

**Date:** 2026-09-11
**Dataset:** BattLeDIM 2018 (Historical Development Dataset)
**Deliverables:**
- Channel Coverage Inventory: [`reports/audit/channel_coverage.csv`](../../reports/audit/channel_coverage.csv)
- Sensor Coverage Profile Diagram: [`reports/audit/sensor_coverage_profile.svg`](../../reports/audit/sensor_coverage_profile.svg)

---

## 1. Sensor Coverage and Completeness Audit

A systematic pre-cleaning audit of all raw 2018 SCADA files was conducted across all 119 channels.

### Summary Metrics
| Metric | Value | Interpretation |
|---|---|---|
| **Total Sensor Channels** | 119 | 3 Flows, 33 Pressures, 82 Demands (AMR), 1 Tank Level |
| **Observation Horizon** | 2018-01-01 00:00:00 to 2018-12-31 23:55:00 | Exactly 365 calendar days |
| **Cadence** | 5 minutes | Uniform 5-minute interval grid |
| **Expected Timestamps per Channel** | 105,120 | 365 days × 288 steps/day |
| **Observed Timestamps per Channel** | 105,120 | 100% temporal completeness |
| **Missing Cells (NaN / Null)** | 0 | 0.00% missing data |
| **Duplicate Timestamps** | 0 | Strictly monotonic increasing |
| **Missing Intervals (> 5 min)** | 0 | Continuous time grid without gaps |
| **Nonfinite Values (inf / -inf)** | 0 | Fully finite numerical observations |
| **Suspicious Physical Ranges** | 0 | All pressures [25.67m, 72.84m], flows >= 0, level [1.0m, 4.88m] |
| **Unresolved Channels** | 0 | 100% mapped to nominal `L-TOWN.inp` elements |

---

## 2. Spatial Coverage vs. Temporal Completeness

While temporal completeness is 100%, spatial instrumentation is highly heterogeneous across the network topology:

1. **Area A (Main distribution network, 657 junctions)**:
   - **Pressures**: 29 sensors (moderate spatial density across major distribution mains).
   - **Inflows**: 2 reservoir lines (`p227` from R1, `p235` from R2).
   - **Demands**: 0 AMR customer meters instrumented.
2. **Area B (Pressure-reduced zone, 31 junctions)**:
   - **Pressures**: 1 sensor (`n226`), downstream of PRV-3 (`n229 -> n226`).
   - **Inflows**: Unmetered direct branch via PRV-3.
   - **Demands**: 0 AMR customer meters instrumented.
3. **Area C (Elevated tank-fed zone, 93 junctions)**:
   - **Pressures**: 3 sensors (`n1`, `n4`, `n31`).
   - **Levels**: 1 tank level sensor (`T1`).
   - **Inflows**: 1 booster pump (`PUMP_1` pumping from `n54` in Area A into `T1`).
   - **Demands**: 82 AMR customer meters (100% of network AMR devices).

> [!NOTE]
> Spatial coverage asymmetry dictates model target design: customer demand forecasting is hydraulically localized to Area C, whereas leak detection observability differs drastically between Area A (pressure sensor rich) and Area B (single pressure sensor).

---

## 3. Target Decision & Event Label Coverage

### Rejection of Aggregate Network Label
The full-year aggregate binary indicator `is_leak` is positive on **97.8025%** of all 2018 timestamps (and 100.0% of October, November, and December timestamps) due to long-running background leaks. A classifier trained on this label achieves >97.8% accuracy simply by outputting a constant positive. It is **strictly rejected as a primary modeling target**.

### Multi-Label Area Vector ($K=3$)
Instead of a single network-wide label or mutually exclusive softmax, we formalize a 3-dimensional binary vector:
$$\mathbf{y}_t = [y_{\text{Area\_A}}, y_{\text{Area\_B}}, y_{\text{Area\_C}}] \in \{0, 1\}^3$$
- **Simultaneous Leaks**: Handled naturally via independent binary cross-entropy with logits.
- **Area A**: 11 events in 2018 (9 onset in Train, 2 in Val_Selection); high sensor observability (29 pressure sensors).
- **Area B**: 1 event in 2018 (`p673`, March 5–23); 1 pressure sensor (`n226`). Validation periods are leak-free, providing clean false-positive stress tests.
- **Area C**: 2 events in 2018 (`p257` long-term, `p31` summer); 82 AMR meters, 3 pressure sensors, 1 pump, 1 tank.

### Demand Forecast Target Contract
- **Target**: Next-hour cumulative consumption across all 82 AMR meters in Area C:
  $$Y_{t+1:t+12} = \left[ \sum_{m=1}^{82} d_{m, t+5\text{m}}, \dots, \sum_{m=1}^{82} d_{m, t+60\text{m}} \right]$$
- **Units**: $\text{m}^3/\text{h}$ (converted from L/h).
- **Evaluation**: Explicitly defined as observed customer metered consumption in Area C, not total system water production.

### Event Label Coverage Provenance
Because the BattLeDIM release does not provide an independent external field repair register to independently verify all unobserved network leaks, event label coverage completeness is formally classified as **Unknown (unsupported)** with label provenance and observed event counts explicitly documented rather than assuming 100% completeness.

---

## 4. Chronological Splits, Boundary Purging & Manifest

### Partition Schedule
All partitions are defined as strictly half-open $[t_{\text{start}}, t_{\text{end}})$ intervals preserving the source's unspecified timezone:

| Partition | Interval | Permitted Use | Origin Timestamps | Operational Eligible | Strict Eligible |
|---|---|---|---|---|---|
| **Train** | `[2018-01-01, 2018-10-01)` | Model and transform fitting only | 78,624 | 78,326 | 2,012 |
| **Val: Model Selection** | `[2018-10-01, 2018-11-01)` | Architecture tuning, early stopping | 8,928 | 8,917 | 0 |
| **Val: Calibration** | `[2018-11-01, 2018-12-01)` | Probability calibration fitting | 8,640 | 8,629 | 0 |
| **Val: Alert Selection** | `[2018-12-01, 2019-01-01)` | Decision threshold / alert tuning | 8,928 | 8,917 | 0 |
| **Final Test** | `[2019-01-01, 2020-01-01)` | Locked final evaluation | 105,120 | Locked | Locked |

### Operational Rolling vs. Strict Event-Independent Protocols
1. **Operational Rolling Protocol**:
   - Reflects true real-time continuous deployment.
   - For prediction origin $t$, 24 hours of preceding history $[t-24\text{h}+5\text{m}, t]$ is observed and valid, even when crossing partition start boundaries (e.g. on Oct 1, late-September data is legitimate historical context).
   - Future targets $[t+5\text{m}, t+60\text{m}]$ are strictly bounded within partition $P$; any window crossing the partition end is truncated (exactly 11 windows per boundary).
   - Total operational eligible windows in 2018: **104,789** (287 initial warmup steps + 44 boundary truncations excluded).

2. **Strict Event-Independent Protocol**:
   - Aims to prevent any temporal leak continuation across partitions.
   - Any window whose historical context or forecast horizon touches a boundary-crossing event (`p257` in Area C; `p427`, `p654`, `p810` in Area A) is completely purged.
   - **Critical Finding & Limitation**: Because `p257` is a chronic background leak running from Jan 9 through year-end, and `p427`/`p654`/`p810` start in summer and run through year-end, applying network-wide strict purging leaves only **2,012 windows** in Train (Jan 1–8) and **0 windows** in the validation blocks!
   - As mandated by the handoff rules, this limitation is documented explicitly rather than silently weakening the purge rule or claiming artificial independence. Operational rolling evaluation is the primary practical benchmark, while strict comparisons must isolate localized non-overlapping burst episodes.
