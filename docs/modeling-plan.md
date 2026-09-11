# Current modeling plan

Revised 2026-09-11: compare all five hybrids using the CSV workflow in the
[root README](../README.md) and [implementation handoff](HANDOFF_README.md).

Required: 1D-CNN + LSTM; CNN + BiLSTM + Attention; Autoencoder + LSTM;
Transformer + CNN; GNN + LSTM.

Use the [paper register](PAPERS_AND_REPRODUCTION.md) to distinguish reproduction from
adaptation, and the [change review](PLAN_CHANGE_REVIEW.md) before reusing existing splits.

No dashboard or full-network graph application is required. M5 retains only the minimum
sensor adjacency needed for a graph operation. Existing audit/mapping utilities remain
useful. The previous core-versus-optional-model ordering, mandatory dual head and fixed
monthly calibration schedule are superseded. This document does not implement models.
