# AI-Based Smart Water Distribution and Leakage Detection

A Python research project for forecasting water demand and identifying potential pipeline leaks from flow, pressure, consumption and time-series signals.

**Status:** dataset pipeline and seasonal baseline implemented. CNN + LSTM, calibrated risk scoring and the application are planned; no trained AI accuracy is claimed. See the [initial verification results](docs/verification.md).

## Dataset recommendation

Choose **BattLeDIM / L-Town as the primary leakage dataset**, and **BWDF as a separate weather-aware demand forecasting benchmark**. These are different networks: do not join their rows or attach BWDF weather to L-Town.

| Priority | Dataset | Best use | Main limitation |
| --- | --- | --- | --- |
| 1 | [BattLeDIM / L-Town](https://doi.org/10.5281/zenodo.4017659) | Joint hydraulic signal analysis, leak detection and localization; includes sensor CSVs, leak time series and an EPANET model | Simulated benchmark; no supplied weather or separate abnormal-consumption class |
| 2 | [Battle of Water Demand Forecasting (BWDF)](https://github.com/WaterFutures/wf4bwdf) | Hourly demand forecasting across 10 district metered areas with weather | Not a labeled pipe-leak benchmark; net inflow includes losses |
| 3 | [LeakDB](https://github.com/KIOS-Research/LeakDB) | Additional synthetic leak scenarios and robustness across network conditions | Simulation-to-field generalization must be tested |
| 4 | [Water demand datasets, Mendeley](https://doi.org/10.17632/4yhprsgjrf.1) | Simple real utility forecasting comparison; choose Dataset 2, hourly Hillsborough inflow/outflow | No pipe-leak labels; Dataset 3 is sewer flow |

The photograph suggests [UCI Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption). It is useful for practicing time-series code, but electricity cannot validate hydraulic leak detection. Indian open-data portals are a discovery route, not a verified substitute for timestamped, labeled pressure/flow measurements.

See [dataset selection and sources](docs/datasets.md) for access, coverage and licensing notes. Sources checked on September 11, 2026.

## Recommended hybrid technique

Start with **1D-CNN + unidirectional LSTM**. CNN layers can extract short local signal patterns; the LSTM can model their temporal context. This is our practical starting hypothesis, not an experimentally proven winner.

Use a shared encoder with a future-demand regression head and a current-leak classification head. First establish seasonal persistence and conventional machine-learning baselines; add attention only if ablation results justify it. Use **GNN + LSTM** as the later topology-aware extension if localization becomes central.

The [modeling plan](docs/modeling-plan.md) compares all five suggested hybrids, specifies chronological evaluation, and defines the steps needed to support Normal / Leakage / Abnormal consumption plus a calibrated leak-risk score. Detecting an existing small leak and predicting a future failure are separate targets.

## Quick start

Python 3.11 or newer. From the project root, in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

# Small start: 2018 flow data, nominal network, and upstream README (~4.4 MB)
.\.venv\Scripts\smart-water.exe download
.\.venv\Scripts\smart-water.exe prepare
.\.venv\Scripts\smart-water.exe baseline

# Add pressure, AMR consumption, tank levels and ground-truth leak labels
.\.venv\Scripts\smart-water.exe download --full
.\.venv\Scripts\smart-water.exe prepare

# Validate the code without network access
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

On Linux/macOS, substitute `.venv/bin/python` and `.venv/bin/smart-water` for the executable paths. Downloading requires internet access. Completed files are checked against the publisher's MD5 checksums; raw data and generated reports are excluded from Git.

The baseline forecasts sensor `p227` flow in m3/h using the value 288 five-minute steps earlier. It reports MAE and RMSE for October–December 2018, using observations available at each rolling forecast origin. This is a **flow forecast sanity check**, not a clean customer-demand estimate, leak detector, or fixed-origin multi-day forecast. Try `--period 2016` for a weekly seasonal comparison.

For final evaluation, fetch 2019 after selecting the model and thresholds on 2018:

```powershell
.\.venv\Scripts\smart-water.exe download --year 2019 --full
.\.venv\Scripts\smart-water.exe prepare --year 2019
.\.venv\Scripts\smart-water.exe baseline --input data/raw/battledim/2019_SCADA_Flows.csv --start 2019-01-02 --output reports/baseline_2019.json
```

This separate-year baseline omits its first day's warmup; a future fixed-origin forecast benchmark must specify its own horizon and allowed history.

## Repository layout

```text
src/smart_water/
  data.py              # Verified downloads, schema/time checks, separate labels
  baseline.py          # Rolling seasonal persistence and forecast metrics
  cli.py               # download / prepare / baseline commands
tests/                 # Units, time integrity, target separation, past-only baseline
docs/
  datasets.md          # Ranked datasets and source references
  modeling-plan.md     # Architecture, labels, experiment design and milestones
data/raw/              # Original datasets (ignored)
data/processed/        # Generated features, labels and audit summary (ignored)
reports/               # Generated metrics (ignored)
.github/workflows/     # Python lint and tests
```

Preparation reads available sensor groups, rejects missing/nonfinite readings and broken or mismatched time grids, converts AMR L/h to m3/h, and adds calendar features. It preserves source timestamps without inventing a timezone. `*_features.csv` never contains leak ground truth; `*_labels.csv` contains a binary indicator when label files exist. The aggregate label means **any leak anywhere in the network**, not the label of an individual pipe or sensor.

The full 2018 audit found that this aggregate label is positive **97.8% of the time**. It is an audit output; design area/pipe targets before training a useful classifier and evaluate event detection rather than timestamp accuracy.

## Next milestones

1. Audit full 2018 data, sensor coverage, leak prevalence and event intervals.
2. Establish forecasting and anomaly-detection baselines with chronological validation.
3. Implement and train CNN + LSTM; evaluate event detection, delay and forecasting error.
4. Calibrate probabilities and add separately labeled abnormal-demand scenarios.
5. Compare attention and GNN variants, then build a dashboard with sensor-level evidence.

This starter covers monitoring research. Pump scheduling, valve control and supply allocation optimization are later work requiring hydraulic and operational constraints.

## Attribution

BattLeDIM: Vrachimis et al., *Dataset of BattLeDIM: Battle of the Leakage Detection and Isolation Methods*, 2020, [DOI 10.5281/zenodo.4017659](https://doi.org/10.5281/zenodo.4017659), CC BY 4.0. Downloads retain the source README and checksum manifest. Other datasets retain their own terms; this repository does not redistribute them. No project software license has been selected yet.
