"""Tests for pipe-level leak event inventory extraction and validation (Step 2 acceptance)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smart_water.audit import parse_network_topology
from smart_water.inventory import extract_leak_events, reconstruct_positive_mask


def test_inventory_hand_computable_fixtures():
    """Verify one-sample, repeated, overlapping, and boundary-censored events with hand-computed volumes."""
    mock_network = {
        "pipes": {
            "p1": {"node1": "n1", "node2": "n2"},
            "p2": {"node1": "n3", "node2": "n4"},
        },
        "pipe_to_area": {"p1": "Area_A", "p2": "Area_B"},
    }

    # 6 time steps: 00:00, 00:05, 00:10, 00:15, 00:20, 00:25
    ts = pd.date_range("2018-01-01 00:00", periods=6, freq="5min")

    # p1: Left-censored at 00:00 (12.0 m3/h), then 0, then 1-sample at 00:15 (24.0 m3/h), then right-censored at 00:25 (6.0 m3/h)
    # p2: Overlapping event from 00:10 to 00:20 (values: 6.0, 18.0, 12.0)
    df = pd.DataFrame(
        {
            "p1": [12.0, 0.0, 0.0, 24.0, 0.0, 6.0],
            "p2": [0.0, 0.0, 6.0, 18.0, 12.0, 0.0],
        },
        index=ts,
    )

    events_df, unresolved_df = extract_leak_events(df, mock_network, threshold=0.0)

    assert unresolved_df.empty
    assert len(events_df) == 4  # 3 on p1, 1 on p2

    # Verify event 1 on p1: Left-censored, 1 sample
    # Volume: 12.0 m3/h * (5/60) h = 1.0 m3. Duration = 5 min = 0.0833 h.
    ev_left = events_df[(events_df["pipe_id"] == "p1") & (events_df["left_censored"])].iloc[0]
    assert ev_left["onset"] == "2018-01-01 00:00:00"
    assert ev_left["end_exclusive"] == "2018-01-01 00:05:00"
    assert ev_left["volume_m3"] == pytest.approx(1.0)
    assert ev_left["duration_hours"] == pytest.approx(5.0 / 60.0)
    assert ev_left["left_censored"] == True
    assert ev_left["right_censored"] == False

    # Verify 1-sample isolated event on p1 at 00:15
    # Volume: 24.0 m3/h * (5/60) h = 2.0 m3.
    ev_mid = events_df[(events_df["pipe_id"] == "p1") & (events_df["onset"] == "2018-01-01 00:15:00")].iloc[0]
    assert ev_mid["onset"] == "2018-01-01 00:15:00"
    assert ev_mid["last_active_sample"] == "2018-01-01 00:15:00"
    assert ev_mid["end_exclusive"] == "2018-01-01 00:20:00"
    assert ev_mid["volume_m3"] == pytest.approx(2.0)
    assert ev_mid["peak_flow_m3_h"] == 24.0

    # Verify right-censored event on p1 at 00:25
    ev_right = events_df[(events_df["pipe_id"] == "p1") & (events_df["right_censored"])].iloc[0]
    assert ev_right["onset"] == "2018-01-01 00:25:00"
    assert ev_right["end_exclusive"] == "2018-01-01 00:30:00"
    assert ev_right["right_censored"] == True
    assert ev_right["volume_m3"] == pytest.approx(0.5)

    # Verify p2 multi-sample overlapping event (00:10 to 00:20)
    # Samples: 6.0, 18.0, 12.0 -> Duration: 3 steps = 15 min = 0.25 h.
    # Volume: (6 + 18 + 12) * (5/60) = 36 * (1/12) = 3.0 m3. Peak: 18.0, Mean: 12.0
    ev_p2 = events_df[events_df["pipe_id"] == "p2"].iloc[0]
    assert ev_p2["onset"] == "2018-01-01 00:10:00"
    assert ev_p2["last_active_sample"] == "2018-01-01 00:20:00"
    assert ev_p2["end_exclusive"] == "2018-01-01 00:25:00"
    assert ev_p2["duration_hours"] == pytest.approx(0.25)
    assert ev_p2["volume_m3"] == pytest.approx(3.0)
    assert ev_p2["peak_flow_m3_h"] == 18.0
    assert ev_p2["mean_flow_m3_h"] == 12.0

    # Test mask reconstruction for p1, p2, and combined
    mask_p1 = reconstruct_positive_mask(events_df, ts, pipe_id="p1")
    assert np.array_equal(mask_p1, df["p1"].to_numpy() > 0)

    mask_p2 = reconstruct_positive_mask(events_df, ts, pipe_id="p2")
    assert np.array_equal(mask_p2, df["p2"].to_numpy() > 0)

    mask_all = reconstruct_positive_mask(events_df, ts)
    assert np.array_equal(mask_all, (df > 0).any(axis=1).to_numpy())


def test_unresolved_pipe_handling():
    """Verify that unmapped pipe IDs are cleanly routed to the unresolved table."""
    mock_network = {"pipes": {}, "pipe_to_area": {}}
    ts = pd.date_range("2018-01-01 00:00", periods=2, freq="5min")
    df = pd.DataFrame({"unknown_pipe_999": [0.0, 5.0]}, index=ts)

    events_df, unresolved_df = extract_leak_events(df, mock_network)
    assert events_df.empty
    assert len(unresolved_df) == 1
    assert unresolved_df.iloc[0]["pipe_id"] == "unknown_pipe_999"
    assert "not found" in unresolved_df.iloc[0]["unresolved_reason"]


def test_unknown_label_makes_adjacent_event_boundaries_incomplete():
    network = {
        "pipes": {"p1": {"node1": "n1", "node2": "n2"}},
        "pipe_to_area": {"p1": "Area_A"},
    }
    ts = pd.date_range("2018-01-01", periods=3, freq="5min")
    frame = pd.DataFrame({"p1": [1.0, np.nan, 1.0]}, index=ts)
    events, _ = extract_leak_events(frame, network)
    assert len(events) == 2
    assert events["incomplete_volume"].all()
    assert events.iloc[0]["unknown_after"]
    assert events.iloc[1]["unknown_before"]


def test_full_2018_leak_event_inventory_and_mask_reconstruction():
    """Verify extraction on official 2018_Leakages.csv against nominal network."""
    leaks_path = Path("data/raw/battledim/2018_Leakages.csv")
    inp_path = Path("data/raw/battledim/L-TOWN.inp")
    if not leaks_path.exists() or not inp_path.exists():
        pytest.skip("2018_Leakages.csv or L-TOWN.inp missing")

    network_info = parse_network_topology(inp_path)
    leaks_df = pd.read_csv(leaks_path, sep=";", decimal=",", index_col="Timestamp")
    leaks_df.index = pd.to_datetime(leaks_df.index)

    events_df, unresolved_df = extract_leak_events(leaks_df, network_info, threshold=0.0)

    # 1. Zero unresolved pipes
    assert unresolved_df.empty, f"Found unresolved pipes: {unresolved_df}"

    # 2. 14 events across the 14 pipes
    assert len(events_df) == 14
    assert set(events_df["pipe_id"]) == set(leaks_df.columns)

    # 3. Exact reconstruction of ground truth mask across all 105,120 timestamps
    reconstructed_mask = reconstruct_positive_mask(events_df, leaks_df.index)
    ground_truth_mask = (leaks_df > 0).any(axis=1).to_numpy()
    assert np.array_equal(reconstructed_mask, ground_truth_mask)

    # 4. Check right-censored events at end of 2018 (p257, p427, p654, p810)
    right_censored = events_df[events_df["right_censored"]]
    assert set(right_censored["pipe_id"]) == {"p257", "p427", "p654", "p810"}
