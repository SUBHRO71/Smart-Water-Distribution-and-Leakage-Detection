# Prerequisite and feasibility execution report

Run date: 2026-09-11.

## BattLeDIM audit result

The original BattLeDIM area targets failed the classification gate. Among 30
area-target/fold combinations, only two assessments contained both classes and only one
also had training-side SMOTE support. Those targets are excluded from the five-model
comparison. The sensor, event, leakage, boundary and missingness audits remain useful
for the separate BattLeDIM demand track.

## LeakDB replacement

The official LeakDB Hanoi_CMH archive was pinned at repository commit
`131144ba423a82639f881adab0d493ab50e3b2fb` (archive MD5
`700e10f8a90f028f838fcae49660225c`). The prepared table contains 175,200 rows across
ten scenarios, 27,091 positive labels, eight positive runs and no missing cells. Constant
pressure Node 1 is excluded, leaving 31 pressure inputs.

The protocol uses Jan-Sep 2017 as development and Oct-Dec as locked test, with ten
leave-one-scenario-out development folds. All ten training folds contain both classes
and support SMOTE with `k_neighbors=5`. Four held-out folds have both classes; aggregate
out-of-fold reporting includes all scenarios and states valid-fold counts.

## Executed preprocessing

Causal windows contain 24 half-hour observations with stride 6. Execution produced
29,170 windows: 21,810 development and 7,360 locked test. Fold 1's scaler was fitted on
training only; SMOTE changed training counts from 16,121 negative / 3,508 positive to
16,121 / 16,121. The 2,181 held-out labels stayed unchanged.

The preprocessing gate is **open**. Conventional baselines are the next phase.
