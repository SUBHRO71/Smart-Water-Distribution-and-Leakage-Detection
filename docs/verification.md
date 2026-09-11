# Starter verification

Run date: 2026-09-11. Local environment: Windows, Python 3.13.7, NumPy 2.5.3,
pandas 2.3.3, pytest 9.1.1, Ruff 0.16.7. Dependency ranges in `pyproject.toml`
support installation; these versions record the actual initial verification environment.

- Nine offline tests passed, including timestamp integrity, decimal-comma parsing,
  unit conversion, label separation, causal baseline alignment, download checksum
  rejection and verified-file reuse.
- Ruff checks passed.
- Official 2018 flow data passed checksum and schema validation: 105,120 rows,
  January 1 at 00:00 through December 31 at 23:55, every five minutes.

## Initial baseline

Command: `smart-water baseline` after `smart-water download`.

| Setting / metric | Value |
| --- | --- |
| Target | p227 flow, m3/h |
| Method | Previous-day seasonal persistence, lag 288 |
| Evaluation | Rolling one-step, October–December 2018 |
| Evaluation rows | 26,496 |
| MAE | 8.063319 m3/h |
| RMSE | 11.307155 m3/h |

This is a reproducible starter sanity check, not a trained hybrid model result,
a leak detection score, a 2019 final test result, or an uncontaminated estimate of
customer demand. Generated JSON and downloaded files remain in ignored local folders.
