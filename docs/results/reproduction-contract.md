# Frozen classification reproduction contract

Run date: 2026-09-11.

The five-model comparison uses LeakDB Hanoi_CMH from `KIOS-Research/LeakDB`, pinned at
commit `131144ba423a82639f881adab0d493ab50e3b2fb`. Archive MD5:
`700e10f8a90f028f838fcae49660225c`.

- Inputs: pressure Node 2 through Node 32; constant Node 1 is excluded.
- Target: binary leak presence at the final timestamp of each window.
- Window: 24 observations at 30-minute cadence, stride 6.
- Development: `[2017-01-01, 2017-10-01)`.
- Locked test: `[2017-10-01, 2018-01-01)`.
- Cross-validation: leave one of ten scenarios out inside development.
- Preprocessing: fit independently on each fold-training set.
- Balancing: compare no-SMOTE with training-only SMOTE (`seed=42`, `k_neighbors=5`).
- Report aggregate out-of-fold metrics and valid-fold counts. Keep test locked until
  model, calibration and alert choices are frozen.

LeakDB is the exact dataset anchor. Architecture papers are references; results are not
called exact reproductions when their scenarios, features or splits differ.
