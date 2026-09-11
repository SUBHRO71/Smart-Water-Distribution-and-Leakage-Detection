"""CSV preparation and scenario-grouped feasibility checks for LeakDB Hanoi."""

import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from smart_water.data import file_md5


def prepare_hanoi_archive(archive: Path, output: Path) -> dict[str, Any]:
    """Combine the ten Hanoi pressure/label scenarios into one audited CSV."""
    frames = []
    scenario_rows = []
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        for scenario in range(1, 11):
            frame = _read_scenario(source, names, scenario)
            frames.append(frame)
            scenario_rows.append(
                {
                    "scenario": scenario,
                    "rows": len(frame),
                    "pressure_channels": len([c for c in frame if c.startswith("pressure__")]),
                    "negative": int((frame["label"] == 0).sum()),
                    "positive": int((frame["label"] == 1).sum()),
                    "events": _count_positive_runs(frame["label"]),
                    "missing_cells": int(frame.isna().sum().sum()),
                }
            )
    combined = pd.concat(frames, ignore_index=True)
    pressure_columns = [name for name in combined if name.startswith("pressure__")]
    constant_pressure_channels = [
        name for name in pressure_columns if combined[name].nunique(dropna=False) == 1
    ]
    output.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output / "hanoi_pressure_labels.csv", index=False)
    audit = pd.DataFrame(scenario_rows)
    audit.to_csv(output / "hanoi_scenario_audit.csv", index=False)
    summary = {
        "source_archive": str(archive),
        "source_md5": file_md5(archive),
        "rows": len(combined),
        "scenarios": len(frames),
        "pressure_channels": 32,
        "usable_pressure_channels": len(pressure_columns) - len(constant_pressure_channels),
        "excluded_constant_pressure_channels": constant_pressure_channels,
        "cadence_minutes": 30,
        "negative": int((combined["label"] == 0).sum()),
        "positive": int((combined["label"] == 1).sum()),
        "events": int(audit["events"].sum()),
        "missing_cells": int(combined.isna().sum().sum()),
    }
    (output / "hanoi_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _read_scenario(
    source: zipfile.ZipFile,
    names: list[str],
    scenario: int,
) -> pd.DataFrame:
    prefix = f"Scenario-{scenario}/"
    labels = pd.read_csv(io.BytesIO(source.read(prefix + "Labels.csv")))
    labels["Timestamp"] = pd.to_datetime(labels["Timestamp"], errors="raise")
    labels = labels.rename(columns={"Timestamp": "timestamp", "Label": "label"})
    pressure_names = sorted(
        [n for n in names if n.startswith(prefix + "Pressures/Node_") and n.endswith(".csv")],
        key=lambda name: int(re.search(r"Node_(\d+)\.csv$", name).group(1)),
    )
    if len(pressure_names) != 32:
        raise ValueError(f"Scenario {scenario}: expected 32 pressure files")
    frame = labels
    for name in pressure_names:
        node = re.search(r"Node_(\d+)\.csv$", name).group(1)
        pressure = pd.read_csv(io.BytesIO(source.read(name)))
        pressure["Timestamp"] = pd.to_datetime(pressure["Timestamp"], errors="raise")
        pressure = pressure.rename(
            columns={"Timestamp": "timestamp", "Value": f"pressure__node_{node}"}
        )
        frame = frame.merge(pressure, on="timestamp", how="inner", validate="one_to_one")
    if len(frame) != len(labels):
        raise ValueError(f"Scenario {scenario}: pressure/label timestamps do not align")
    if frame["timestamp"].duplicated().any() or not frame["timestamp"].is_monotonic_increasing:
        raise ValueError(f"Scenario {scenario}: timestamps must be unique and increasing")
    if len(frame) > 1 and not frame["timestamp"].diff().iloc[1:].eq(pd.Timedelta(30, unit="m")).all():
        raise ValueError(f"Scenario {scenario}: expected complete 30-minute cadence")
    if set(frame["label"].dropna().unique()) - {0.0, 1.0}:
        raise ValueError(f"Scenario {scenario}: labels must be binary")
    values = frame.drop(columns=["timestamp", "label"]).to_numpy(dtype=float)
    if not np.isfinite(values).all() or frame["label"].isna().any():
        raise ValueError(f"Scenario {scenario}: missing/nonfinite values")
    frame.insert(0, "scenario", scenario)
    return frame


def scenario_fold_feasibility(
    frame: pd.DataFrame,
    split_time: str = "2017-10-01",
    smote_k_neighbors: int = 5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Leave one scenario out within development; reserve later time as final test."""
    boundary = pd.Timestamp(split_time)
    development = frame[frame["timestamp"] < boundary]
    test = frame[frame["timestamp"] >= boundary]
    scenarios = sorted(frame["scenario"].unique())
    rows = []
    for fold, held_out in enumerate(scenarios, start=1):
        training = development[development["scenario"] != held_out]
        assessment = development[development["scenario"] == held_out]
        train_neg = int((training["label"] == 0).sum())
        train_pos = int((training["label"] == 1).sum())
        assess_neg = int((assessment["label"] == 0).sum())
        assess_pos = int((assessment["label"] == 1).sum())
        minority = min(train_neg, train_pos)
        rows.append(
            {
                "fold": fold,
                "held_out_scenario": int(held_out),
                "train_negative": train_neg,
                "train_positive": train_pos,
                "assessment_negative": assess_neg,
                "assessment_positive": assess_pos,
                "assessment_events": _count_positive_runs(assessment["label"]),
                "smote_supported": train_neg > 0 and train_pos > 0 and minority > smote_k_neighbors,
                "assessment_has_both_classes": assess_neg > 0 and assess_pos > 0,
            }
        )
    fold_report = pd.DataFrame(rows)
    test_by_scenario = []
    for scenario in scenarios:
        before = development[development["scenario"] == scenario]["label"]
        after = test[test["scenario"] == scenario]["label"]
        test_by_scenario.append(
            {
                "scenario": int(scenario),
                "negative": int((after == 0).sum()),
                "positive": int((after == 1).sum()),
                "events": _count_positive_runs(after),
                "positive_carryover_at_boundary": bool(
                    len(before) and len(after) and before.iloc[-1] == 1 and after.iloc[0] == 1
                ),
            }
        )
    summary = {
        "split_time": str(boundary),
        "cv": "10-fold leave-one-scenario-out within chronological development period",
        "all_training_folds_smote_supported": bool(fold_report["smote_supported"].all()),
        "assessment_folds_with_both_classes": int(fold_report["assessment_has_both_classes"].sum()),
        "oof_negative": int(fold_report["assessment_negative"].sum()),
        "oof_positive": int(fold_report["assessment_positive"].sum()),
        "test_negative": int((test["label"] == 0).sum()),
        "test_positive": int((test["label"] == 1).sum()),
        "test_scenarios": test_by_scenario,
    }
    return fold_report, summary


def _count_positive_runs(labels: pd.Series) -> int:
    values = labels.fillna(0).astype(int).to_numpy()
    if not len(values):
        return 0
    return int(((values == 1) & np.r_[True, values[:-1] != 1]).sum())
