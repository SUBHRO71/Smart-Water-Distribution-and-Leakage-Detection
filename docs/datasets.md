# Dataset selection

Research date: 2026-09-11. Ranking reflects this project's combined requirements, not a universal benchmark ranking.

Current execution decision (2026-09-12): LeakDB Hanoi_CMH is prepared for the common
five-model comparison. Each paper experiment must use that paper's actual data and
features; see the [main README](../README.md). The original ranking below is historical
dataset research, not permission to substitute one dataset for another paper's data.

## 1. BattLeDIM / L-Town — original choice and audit source

Use the [versioned Zenodo release](https://doi.org/10.5281/zenodo.4017659), the [organizer repository](https://github.com/KIOS-Research/BattLeDIM), and its [problem/rules document](https://zenodo.org/records/3902046).

The release provides two years of simulated hydraulic time series: 2018 for development and 2019 for evaluation, at five-minute resolution. Sensor channels include 33 pressures, 3 flows, 1 tank level and 82 AMR readings. Published metadata define flow and leak units as m3/h, AMR demand as L/h, and pressure/level as meters. The loader converts AMR demand to m3/h. [Source metadata](https://zenodo.org/records/4017659/files/README.txt?download=1).

Download sensor CSVs, separate leak labels and the nominal `L-TOWN.inp`. Do not use `L-TOWN_Real.inp` or hidden leak configuration as predictors. Retrospective full labels support a supervised research experiment; distinguish that from the original challenge's limited historical repair information. No weather, authentic geographic weather join, or explicit non-leak abnormal-consumption class is supplied. Network geometry supports topology experiments; it is not grounds for attaching arbitrary city weather.

Access: direct public downloads, no account required. License: CC BY 4.0, verified from the [record API](https://zenodo.org/api/records/4017659). The starter downloads about 4.4 MB; a full single-year sensor/label set is about 92 MB, avoiding duplicate XLSX copies.

## 2. BWDF — best complementary forecasting dataset

BWDF covers hourly net inflow for ten DMAs and accompanying temperature, rainfall, humidity and wind observations. Its forecasting task spans a week. Competition weather for the forecast week is treated as perfect information; a deployment evaluation must instead use weather forecasts available at the origin. Net inflow includes consumption and losses, so label the prediction as DMA net inflow. [Participant methods and data description](https://www.mdpi.com/2673-4591/69/1/60).

Access: [Water Futures implementation](https://github.com/WaterFutures/wf4bwdf) and its [versioned software archive](https://doi.org/10.5281/zenodo.17185809) document loading the original paper's supplementary datasets. The organizer's original conference URL did not open during this research. This is a verified candidate, not an integrated or downloaded dataset in this starter. Confirm original supplementary-data terms before reuse; an implementation's software license does not establish the data license.

Train and evaluate it separately from BattLeDIM. It does not supply the pipe-leak labels and network sensing needed for the primary classification task.

## 3. LeakDB — robustness and synthetic scenario expansion

The [creator repository](https://github.com/KIOS-Research/LeakDB) links the complete benchmark and example code. It contains artificially generated, realistic leak scenarios under varying network conditions. Use it for sensitivity to leak size and operating conditions, and for testing cross-network transfer. Its original examples/scoring use MATLAB; Python adaptation is additional work. [Original paper](https://zenodo.org/records/1313116).

Follow the current complete-dataset link from the creator repository and verify that release's data license separately from code. Do not pool highly related simulation windows randomly across train/test: hold out complete scenarios and, where feasible, entire networks. Synthetic performance alone cannot establish field reliability.

## 4. Mendeley Water demand datasets — smaller forecasting alternative

[Jinduan Chen, version 1, DOI 10.17632/4yhprsgjrf.1](https://data.mendeley.com/datasets/4yhprsgjrf/1) is released under CC BY 4.0. Dataset 2 contains hourly inflow/outflow at production and storage facilities in Hillsborough County, Florida, April–December 2012. Dataset 3 contains sewer flows, so it is not the drinking-water choice. Useful for a compact forecast benchmark; no documented pressure/network leak labels for the requested classifier.

## Why not electricity or generic Indian water statistics?

[UCI household electricity consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) measures electrical quantities. It can exercise preprocessing and sequence models but provides no evidence of hydraulic leakage. Broad water availability or annual supply statistics cannot replace synchronized SCADA. An Indian utility pilot would need permissioned time series, units, sensor locations/topology, weather matched to actual location/time, and independently verified incident labels. No specific Indian dataset meeting that full contract was verified in this search.

## Data acquisition policy

Keep originals immutable; retain attribution and release identifiers. The downloader records upstream checksums and license metadata. Validate timestamps, cadence, units, missingness, sensor mapping and events before training. Keep test labels out of features and model selection. Document additional data sources and licenses before expanding the registry or redistributing files.
