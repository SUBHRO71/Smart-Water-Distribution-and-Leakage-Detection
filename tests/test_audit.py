"""Tests for raw sensor coverage audit and network topology mapping (Step 1 acceptance)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smart_water.audit import audit_raw_dataset, audit_time_series, parse_network_topology


def test_audit_catches_missing_timestamp_value_duplicate_and_stuck_channel():
    """Verify that synthetic anomalies (missing timestamp, missing value, duplicate, stuck) are caught."""
    # 1. Channel with missing value (NaN)
    ts_clean = pd.date_range("2018-01-01 00:00", periods=5, freq="5min")
    vals_missing = pd.Series([10.0, np.nan, 12.0, 11.5, 10.8])
    rec_missing = audit_time_series(ts_clean.to_series(), vals_missing, "test_missing", "Flow", "m3/h")
    assert rec_missing["missing_cells"] == 1
    assert rec_missing["expected_timestamps"] == 5
    assert rec_missing["observed_timestamps"] == 5

    # 2. Channel with missing timestamp (gap > 5 min)
    ts_gap = pd.Series([
        pd.Timestamp("2018-01-01 00:00"),
        pd.Timestamp("2018-01-01 00:05"),
        pd.Timestamp("2018-01-01 00:25"),  # 20 min gap (missing 3 steps)
        pd.Timestamp("2018-01-01 00:30"),
    ])
    vals_normal = pd.Series([1.0, 2.0, 3.0, 4.0])
    rec_gap = audit_time_series(ts_gap, vals_normal, "test_gap", "Pressure", "m")
    assert rec_gap["missing_intervals"] == 1
    assert rec_gap["longest_gap_minutes"] == 20.0
    assert rec_gap["expected_timestamps"] == 7  # 00:00 to 00:30 inclusive has 7 points
    assert rec_gap["observed_timestamps"] == 4

    # 3. Channel with duplicate timestamps
    ts_dup = pd.Series([
        pd.Timestamp("2018-01-01 00:00"),
        pd.Timestamp("2018-01-01 00:05"),
        pd.Timestamp("2018-01-01 00:05"),  # duplicate
        pd.Timestamp("2018-01-01 00:10"),
    ])
    rec_dup = audit_time_series(ts_dup, vals_normal, "test_dup", "Demand", "L/h")
    assert rec_dup["duplicate_timestamps"] == 1

    # 4. Stuck channel (constant run)
    ts_stuck = pd.date_range("2018-01-01 00:00", periods=10, freq="5min")
    vals_stuck = pd.Series([42.5] * 10)
    rec_stuck = audit_time_series(ts_stuck.to_series(), vals_stuck, "test_stuck", "Pressure", "m")
    assert rec_stuck["max_constant_run_steps"] == 10

    # 5. Suspicious physical range
    vals_neg = pd.Series([-5.0, 10.0, 12.0])
    ts_short = pd.date_range("2018-01-01 00:00", periods=3, freq="5min")
    rec_neg = audit_time_series(ts_short.to_series(), vals_neg, "test_neg", "Pressure", "m")
    assert rec_neg["suspicious_range"] is True
    assert "negative_pressure" in rec_neg["suspicious_reasons"]


def test_topology_partitioning_and_channel_mapping():
    """Verify that L-TOWN.inp partitions into Area A, B, C and maps all sensors with 0 unresolved."""
    inp_path = Path("data/raw/battledim/L-TOWN.inp")
    raw_dir = Path("data/raw/battledim")
    if not inp_path.exists() or not (raw_dir / "2018_SCADA_Flows.csv").exists():
        pytest.skip("Full BattLeDIM 2018 dataset not found in data/raw/battledim")

    topology = parse_network_topology(inp_path)
    assert len(topology["junctions"]) == 782
    assert len(topology["pipes"]) == 905
    assert len(topology["pumps"]) == 1
    assert len(topology["valves"]) == 3

    # Areas: Area C must contain T1, Area B must contain n226, Area A contains main network
    assert topology["node_to_area"]["T1"] == "Area_C"
    assert topology["node_to_area"]["n226"] == "Area_B"
    assert topology["node_to_area"]["n300"] == "Area_A"

    audit_df, summary = audit_raw_dataset(raw_dir, inp_path)
    assert summary["total_channels"] == 119
    assert summary["unresolved_channels"] == 0
    assert summary["total_missing_cells"] == 0
    assert summary["total_duplicate_timestamps"] == 0
    assert summary["total_missing_intervals"] == 0

    # Reconcile counts against expected 5-min grid: exactly 105,120 rows for 2018
    for _, row in audit_df.iterrows():
        assert row["observed_timestamps"] == 105120
        assert row["expected_timestamps"] == 105120
        assert row["mapping_status"] == "Mapped"
        assert row["area"] in ("Area_A", "Area_B", "Area_C", "Area_A->Area_C", "Area_A")
