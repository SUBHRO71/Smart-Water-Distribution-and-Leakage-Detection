# Papers and reproduction register

Sources checked on 2026-09-11. These are real publications, not a claim that their
results have been reproduced here. This focused search did not verify one public study
covering all five exact hybrids under the requested SMOTE/ten-fold workflow.

## Recommended reading and suitability

### P1. Benchmark anchor: BattLeDIM

Vrachimis et al. (2022), **Battle of the Leakage Detection and Isolation Methods**,
Journal of Water Resources Planning and Management 148(12), 04022068.
[DOI](https://doi.org/10.1061/(ASCE)WR.1943-5452.0001601),
[organizer description](https://battledim.ucy.ac.cy/),
[dataset](https://doi.org/10.5281/zenodo.4017659).

This anchors the current dataset and challenge protocol: historical 2018 measurements,
2019 evaluation and leak detection/localization. It is a benchmark paper, not a recipe
for the five requested hybrids. Match the applicable challenge information and scoring
before comparing to its results; a new supervised accuracy table is a different experiment.
Data access is verified and the 2018 release was already downloaded.

### P2. Best related autoencoder study; training data differ from our CSVs

Doss, Rokstad and Tscheikner-Gratl (2024), **The performance of encoder–decoder neural
networks for leak detection in water distribution networks**, Water Supply 24(8), 2750–2764.
[Publisher / DOI](https://doi.org/10.2166/ws.2024.174).

Verified recipe: pressure-only AE, VAE and LSTM-AE; leak-free nominal L-Town simulations
generated with WNTR, 80/20 train-validation division, z-standardization and 144-sample
(12-hour) sequences. Evaluation uses separately generated single-leak scenarios.
Pressure reconstruction errors feed change detection. This is not simply training on
the released leaky 2018 CSV. Architecture details are in Supplementary Appendix 1.

Reproduction needs those scenario outputs or a verified generator and the supplement.
Under CSV-only scope, obtain equivalent author-generated CSVs before claiming exact
reproduction; do not add a hydraulic simulator just to hide the mismatch. A pressure-only
M3 adaptation on our data remains useful but is not the published experiment. Do not
copy unrelated sensor-count statements over the actual downloaded release's schema.

### P3. Graph reference; not an exact GNN + LSTM recipe

Garðarsson, Boem and Toni (2022), **Graph-Based Learning for Leak Detection and Localisation
in Water Distribution Networks**, IFAC-PapersOnLine 55(6), 661–666.
[University-hosted full text](https://discovery.ucl.ac.uk/id/eprint/10155676/1/1-s2.0-S2405896322005882-main.pdf).

Uses two graph networks to reconstruct/predict pressure, nominal-model training and
2019 BattLeDIM evaluation. Inputs include 33 pressure sensors; the selected predictor
history is three five-minute samples, and pressure normalization is to [0,1]. Detection
uses residuals mapped to network edges. These details are in Sections 2–3.

A small correlation-based sensor GNN + LSTM does not reproduce its physical graph,
training generation or localization metric. Use this as graph-method background; exact
replication would exceed the present scope unless prepared graph/training artifacts
are obtained. No verified author code link was found in the inspected full text.

### P4. Transformer reference with accuracy/F1; different networks

Luo, Wang, Yang and Zhong (2024), **A Transformer-Based Approach to Leakage Detection in
Water Distribution Networks**, Sensors 24(19), 6294.
[Publisher / DOI](https://doi.org/10.3390/s24196294),
[full-text mirror](https://pmc.ncbi.nlm.nih.gov/articles/PMC11478714/).

Studies simulated pressure data from Hanoi, Net1 and Anytown. The method reconstructs
no-leak pressure, incorporates positional information and uses reconstruction loss to
identify leaks; accuracy and F1 are reported. The paper describes scenario normalization.
It is not a demonstrated reproduction of our Transformer + CNN on BattLeDIM.

Exact scenario files, normalization scope, split and complete hyperparameters still need
verification before selecting it as the primary replication. Confirm whether normalization
uses whole evaluation scenarios; if so, distinguish that offline protocol from our
past-only experiment. Source access was intermittent; no exact training bundle was
downloaded in this review. Do not assume same-name network data equal the authors' data.

### P5. CNN–LSTM–attention reference; acoustic data on request

Umer, Ullah and Kim (2026), **Attention mechanism-based CNN-LSTM hybrid deep learning
model for industrial pipeline leak detection**, Scientific Reports 16, 22763.
[Publisher / DOI](https://doi.org/10.1038/s41598-026-53776-x).

Uses industrial acoustic-emission waveforms and continuous-wavelet representations,
CNN/LSTM feature extraction and multi-head attention. It reports 96.88% classification
accuracy; this is an author-reported acoustic result, not a target or baseline for our
SCADA dataset. Its data-availability section says data are obtainable from the corresponding
author on reasonable request. No request was sent.

Useful architectural background for M1/M2, but it does not establish the exact requested
CNN + BiLSTM + Attention recipe. Full reproduction is blocked until the original data
and feature construction are available. Replacing acoustic/CWT inputs with pressure
CSV columns is a new experiment, not equivalent feature training.

## What to choose now

Follow the [main README](../README.md) for the single execution plan. Verify the actual
paper datasets and recipes before training. LeakDB is prepared for the common comparison;
it does not replace P2's generated L-Town scenarios or P5's acoustic data. Reproduce each
paper's documented split and add mandatory ten-fold CV within development. P1 supplies
benchmark context and has no direct M1 architecture mapping. All five hybrids remain
required; adaptations must be distinguished from exact paper reproductions. The proposed
research gap still requires literature evidence and a controlled experiment.

| Requested model | Verified related reference | Exact model/data equivalence |
| --- | --- | --- |
| 1D-CNN + LSTM | P5 | Different acoustic features; not verified |
| CNN + BiLSTM + Attention | P5 | Attention/CNN/LSTM precedent; exact BiLSTM variant not verified |
| Autoencoder + LSTM | P2 | Related LSTM-AE; generated training/scenario data must match |
| Transformer + CNN | P4 | Related Transformer study; exact hybrid not verified |
| GNN + LSTM | P3 | Related GNN study; exact hybrid and graph differ |

## Required replication worksheet

The implementation model must fill this before calling any result a reproduction:

| Field | Required evidence |
| --- | --- |
| Paper/version | DOI, relevant section/table and supplement/code version |
| Dataset | Exact archive/version/checksum, scenario IDs and generation provenance |
| Inputs | Channel names/order, units, cadence, selection and feature equations |
| Windows | Length, stride, overlap, forecast origin/horizon |
| Labels | Class definitions, event threshold, unknown/censoring policy |
| Preprocessing | Fit population for each transform, filtering and missingness policy |
| Training | Architecture, losses, optimizer, epochs/early stopping, seeds |
| Validation | Exact split/group rules and tuning/calibration access |
| Balancing | Whether the paper uses SMOTE; if newly added, mark adaptation |
| Evaluation | Metric averaging, threshold/event matching, exclusions and test access |
| Outcome | Published result, reproduced result, uncertainty and deviations separately |

Keep incompatible results in separate tables. Adding SMOTE, changing features, using
new labels or replacing the original split with ten-fold CV changes the experimental
protocol. Reproduce first if feasible, then label modifications as the comparative study.

## Workflow references

- [Original SMOTE paper](https://arxiv.org/abs/1106.1813): Chawla et al.,
  *SMOTE: Synthetic Minority Over-sampling Technique* (2002).
- [imbalanced-learn pitfalls](https://imbalanced-learn.org/stable/common_pitfalls.html):
  resampling before splitting contaminates evaluation; keep it inside fold-training.
- [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html):
  time-ordered splits; additional window/event purging is project-specific work.
