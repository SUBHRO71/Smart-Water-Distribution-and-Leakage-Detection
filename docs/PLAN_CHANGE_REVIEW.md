# Existing-work review for the revised plan

Reviewed 2026-09-11. This was a documentation and targeted correctness review, not full
scientific validation of Phase 1. No training or broad implementation was performed.

## Decision: retain useful work; revise contracts before training

| Existing work | Decision | Reason |
| --- | --- | --- |
| data.py, downloader, raw CSVs, checksum metadata | Keep | Directly support CSV workflow |
| audit.py and sensor/pipe mapping | Keep | Coverage and provenance remain useful; a parser is not a full graph application |
| inventory.py and event tables | Keep, correct unknown handling | Event-aware evaluation still needs incident identity |
| targets.py / targets.json | Revalidate, do not treat as frozen | Existing area targets remain constant in validation; ordinary SMOTE is not multi-label |
| splits.py / splits.json / window manifest | Correct and regenerate | Boundary bug, no ten-fold CV, changed selection protocol |
| baseline.py and recorded p227 score | Keep as sanity check | Flow-regression metric is not five-model leak accuracy |
| Phase 1 reports and execution log | Preserve as historical | Old gate decisions/counts do not certify the revised protocol |
| Old optional-model ordering, mandatory dual head, dashboard milestone | Superseded in docs | All five models now required; no dashboard/full-network application |
| .gitignore and local pytest configuration edits | Preserve; improve temp handling later | Data exclusion is useful; fixed temp ownership is brittle |

No model/dashboard/full-network application exists to revert. Do not delete the working
audit or nominal-network parser. Do not reset unrelated local changes. New training
cannot start simply because the previous execution log marked the audit gate complete.

## Confirmed pre-training issues

1. **Partition boundary leak.** In src/smart_water/splits.py, target_end is the last
   included future observation, but crossing is tested with `tgt_end > p_end`. For a
   half-open partition, equality is already outside: use `>=` under that representation.
   tests/test_splits.py explicitly expects equality to be allowed, so its passing result
   misses the defect. The 2018-09-30 23:00 origin includes the October 1 00:00 target.
   Fix the test/implementation together and regenerate counts; do not reuse the reported
   eleven-window boundary exclusions. Also audit window/event intersection endpoint
   conventions and reject target timestamps absent from the available data.

2. **Area labels do not solve classification feasibility.** Local
   reports/audit/label_balance.csv shows A/C=100% positive and B=0% for every October,
   November and December block. Neither calibration nor a useful class-discrimination
   comparison follows from these targets. Retain this finding; redesign targets or choose
   enough independent labeled scenarios before SMOTE/CV. A 14-event inventory cannot be
   enlarged into more independent events by making overlapping windows.

3. **Unknown label samples become false negatives.** A direct fixture `[1, NaN, 1]`
   passed to extract_leak_events produces two runs, both marked incomplete_volume=False;
   create_target_matrix assigns zero at the missing middle timestamp. Unknown flags are
   checked only inside positive runs, which cannot contain NaN under the current logic.
   Preserve missingness and boundary uncertainty, with explicit masks for training/scoring.
   The verified complete 2018 data are unaffected by this fixture, but the claimed generic
   missing-label support is not implemented correctly.

4. **Strict purging is infeasible as the main current benchmark.** Existing reports
   show zero strict validation windows after network-wide continuing-event exclusion.
   Keep this as a limitation, not a passing independent-evaluation gate. Use documented
   target-local/event/scenario feasibility analysis; do not silently weaken independence
   or silently replace chronological folds with shuffled folds.

5. **Metric/wording correction.** The target described as “cumulative consumption” is
   a sequence of sums across AMR meters in m3/h. That is aggregate flow rate at each step,
   not cumulative volume. Update wording or explicitly integrate with the time interval
   if volume is intended. Keep classification and forecast metrics separate.

## Verification performed

- Ruff: passed on the current workspace.
- Default pytest invocation: 12 passed, 8 setup errors caused by the existing fixed
  pytest_tmp folder's permissions; not eight model/data failures.
- With a fresh unique temporary directory: **20 tests passed**. Some tests exercise local
  data and may be skipped on a fresh clone without those files. Passing tests do not
  invalidate the boundary and unknown-label counterexamples above.
- Reviewed local manifests, label support and targeted source code; did not regenerate
  the full audit, train models or claim all Phase 1 acceptance conditions are satisfied.

## Ownership and next action

At review start, .gitignore, pyproject.toml and cli.py were modified locally. Audit,
inventory, target/split modules, tests, configs, docs/results and EXECUTION_README.md
were untracked local work. This documentation update preserves them and does not stage
them as part of the plan commit.

The split-boundary, complete-window, missing-label and demand-wording corrections were
implemented after this review. The ten-fold audit then confirmed that current area labels
do not support the requested experiment. See the
[executed feasibility report](results/prerequisite-and-feasibility.md). The remaining
dependency is a scenario dataset/target with sufficient independent class support.
