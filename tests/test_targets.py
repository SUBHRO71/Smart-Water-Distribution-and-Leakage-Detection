"""Tests for target contract, label balance, and simultaneous leak representation (Step 3 acceptance)."""

from pathlib import Path

import pandas as pd
import pytest

from smart_water.targets import (
    build_targets_config,
    build_unknown_by_area,
    compute_label_balance,
    create_target_matrix,
)


def test_simultaneous_leaks_and_multilabel_independence():
    """Verify that simultaneous leaks across different areas produce independent binary positives."""
    ts = pd.date_range("2018-01-01 00:00", periods=5, freq="5min")
    mock_events = pd.DataFrame([
        {
            "event_id": "EV_001",
            "pipe_id": "p1",
            "area": "Area_A",
            "onset": "2018-01-01 00:05:00",
            "end_exclusive": "2018-01-01 00:20:00",
        },
        {
            "event_id": "EV_002",
            "pipe_id": "p2",
            "area": "Area_C",
            "onset": "2018-01-01 00:10:00",
            "end_exclusive": "2018-01-01 00:25:00",
        },
    ])

    mat = create_target_matrix(ts, mock_events)
    assert mat.shape == (5, 3)
    assert list(mat.columns) == ["leak_Area_A", "leak_Area_B", "leak_Area_C"]

    # 00:00: No leaks
    assert (mat.iloc[0] == [0, 0, 0]).all()
    # 00:05: Area A only
    assert (mat.iloc[1] == [1, 0, 0]).all()
    # 00:10: Area A AND Area C simultaneously active -> [1, 0, 1]
    assert (mat.iloc[2] == [1, 0, 1]).all()
    # 00:15: Area A AND Area C simultaneously active -> [1, 0, 1]
    assert (mat.iloc[3] == [1, 0, 1]).all()
    # 00:20: Area C only (Area A ended at 00:20 exclusive)
    assert (mat.iloc[4] == [0, 0, 1]).all()


def test_targets_contract_specification():
    """Verify that configs/targets.json conforms to the research contract."""
    mock_amrs = [f"n{i}" for i in range(1, 83)]
    cfg = build_targets_config(mock_amrs)

    # Demand target contract
    demand = cfg["demand_target"]
    assert demand["sensor_count"] == 82
    assert demand["horizon_steps"] == 12
    assert demand["unit"] == "m3/h"
    assert demand["loss_function"] == "HuberLoss"

    # Leak target contract
    leak = cfg["leak_targets"]
    assert leak["dimension_K"] == 3
    assert leak["activation"] == "sigmoid"
    assert leak["simultaneous_leaks_permitted"] is True
    assert leak["target_vector"] == ["leak_Area_A", "leak_Area_B", "leak_Area_C"]
    assert "strictly rejected" in leak["aggregate_network_label_policy"]


def test_unknown_pipe_label_is_not_converted_to_negative_area_label():
    ts = pd.date_range("2018-01-01", periods=3, freq="5min")
    leaks = pd.DataFrame({"p1": [1.0, float("nan"), 1.0]}, index=ts)
    events = pd.DataFrame([
        {"area": "Area_A", "onset": ts[0], "end_exclusive": ts[1]},
        {"area": "Area_A", "onset": ts[2], "end_exclusive": ts[2] + pd.Timedelta(5, unit="m")},
    ])
    unknown = build_unknown_by_area(leaks, {"p1": "Area_A"})
    targets = create_target_matrix(ts, events, unknown)
    assert targets.loc[ts[0], "leak_Area_A"] == 1
    assert pd.isna(targets.loc[ts[1], "leak_Area_A"])
    assert targets.loc[ts[2], "leak_Area_A"] == 1


def test_label_balance_on_2018_splits():
    """Verify label balance calculation on official 2018 dataset across partitions."""
    leaks_path = Path("data/raw/battledim/2018_Leakages.csv")
    events_path = Path("reports/audit/leak_events.csv")
    if not leaks_path.exists() or not events_path.exists():
        pytest.skip("2018_Leakages.csv or reports/audit/leak_events.csv missing")

    leaks_df = pd.read_csv(leaks_path, sep=";", decimal=",", index_col="Timestamp")
    leaks_df.index = pd.to_datetime(leaks_df.index)
    events_df = pd.read_csv(events_path)

    partitions = {
        "Train": ("2018-01-01", "2018-10-01"),
        "Val_Selection": ("2018-10-01", "2018-11-01"),
        "Val_Calibration": ("2018-11-01", "2018-12-01"),
        "Val_Alert": ("2018-12-01", "2019-01-01"),
    }

    balance_df = compute_label_balance(events_df, leaks_df, partitions)
    assert not balance_df.empty

    # Verify Area B is NOT constantly positive: only active in March (Train partition)
    val_area_b = balance_df[(balance_df["partition"] != "Train") & (balance_df["target_scope"] == "Area_B")]
    for _, row in val_area_b.iterrows():
        assert row["positive_timestamps"] == 0
        assert row["timestamp_prevalence_pct"] == 0.0

    # Verify Area A has distinct events across partitions
    train_area_a = balance_df[(balance_df["partition"] == "Train") & (balance_df["target_scope"] == "Area_A")].iloc[0]
    assert train_area_a["positive_timestamps"] > 0
    assert train_area_a["active_events_in_partition"] > 0

    # Verify event label coverage completeness is explicitly documented as Unknown
    for _, row in balance_df.iterrows():
        assert "Unknown" in row["event_label_coverage_completeness"]
