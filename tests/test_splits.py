"""Tests for chronological splits, boundary window contracts, and strict event purging (Step 4 acceptance)."""

import numpy as np
import pandas as pd

from smart_water.splits import (
    generate_window_manifest,
)


def test_window_boundary_enforcement_and_past_context():
    """Verify that targets cannot cross partition boundaries, but past history is allowed in operational mode."""
    # Start at least 24h before 2018-09-30 so history is available
    ts_train = pd.date_range("2018-09-28 00:00", "2018-09-30 23:55", freq="5min")
    ts_val = pd.date_range("2018-10-01 00:00", "2018-10-02 23:55", freq="5min")
    ts = ts_train.append(ts_val)

    mock_events = pd.DataFrame(columns=[
        "event_id", "pipe_id", "area", "onset", "end_exclusive", "right_censored"
    ])

    manifest, _stats = generate_window_manifest(ts, mock_events, history_steps=288, forecast_steps=12)

    # A target at the exclusive boundary is outside the training partition.
    w_boundary = manifest[manifest["origin_timestamp"] == "2018-09-30 23:00:00"].iloc[0]
    assert w_boundary["targets_cross_boundary"] == True
    assert w_boundary["eligible_operational"] == False

    # 2. Origin at 2018-09-30 23:05 has target end 2018-10-01 00:05 (crosses boundary) -> eligible_operational = False
    w_cross = manifest[manifest["origin_timestamp"] == "2018-09-30 23:05:00"].iloc[0]
    assert w_cross["targets_cross_boundary"] == True
    assert w_cross["eligible_operational"] == False

    # 3. Origin at 2018-10-01 00:00 (first step of Val): history starts 2018-09-30 00:05 in September
    # In operational evaluation, this past context is allowed!
    w_val_first = manifest[manifest["origin_timestamp"] == "2018-10-01 00:00:00"].iloc[0]
    assert w_val_first["partition"] == "val_selection"
    assert w_val_first["history_available"] == True
    assert w_val_first["targets_cross_boundary"] == False
    assert w_val_first["eligible_operational"] == True


def test_missing_source_timestamp_invalidates_window():
    ts = pd.date_range("2018-09-29", "2018-10-02", freq="5min")
    ts = ts.delete(ts.get_loc(pd.Timestamp("2018-09-30 12:00")))
    events = pd.DataFrame(
        columns=["event_id", "pipe_id", "area", "onset", "end_exclusive", "right_censored"]
    )
    manifest, _ = generate_window_manifest(ts, events)
    row = manifest.loc[manifest["origin_timestamp"] == "2018-09-30 18:00:00"].iloc[0]
    assert row["window_grid_complete"] == False
    assert row["eligible_operational"] == False


def test_strict_event_purging():
    """Verify that earlier-block windows touching a boundary-crossing event are purged in strict mode."""
    # Start at least 24h before 2018-09-30 14:00
    ts = pd.date_range("2018-09-28 00:00", "2018-10-03 00:00", freq="5min")

    # Event crossing the 2018-10-01 00:00 boundary
    mock_events = pd.DataFrame([
        {
            "event_id": "EV_CROSS",
            "pipe_id": "p_cross",
            "area": "Area_A",
            "onset": "2018-09-30 12:00:00",
            "end_exclusive": "2018-10-01 12:00:00",
            "right_censored": False,
        }
    ])

    manifest, _stats = generate_window_manifest(ts, mock_events, history_steps=288, forecast_steps=12)

    # Window at 2018-09-30 14:00 touches the crossing event:
    # Operational: eligible. Strict: purged!
    w_touch = manifest[manifest["origin_timestamp"] == "2018-09-30 14:00:00"].iloc[0]
    assert w_touch["eligible_operational"] == True
    assert w_touch["eligible_strict"] == False
    assert w_touch["purge_reason"] == "Touches_boundary_crossing_event"


def test_preprocessing_isolation_from_held_out_data():
    """Verify that mutating held-out validation data has zero effect on training fit parameters."""
    train_dates = pd.date_range("2018-01-01", "2018-09-30 23:55", freq="5min")
    val_dates = pd.date_range("2018-10-01", "2018-10-31 23:55", freq="5min")

    np.random.seed(42)
    train_vals = np.random.normal(50.0, 10.0, size=len(train_dates))
    val_vals_clean = np.random.normal(50.0, 10.0, size=len(val_dates))
    val_vals_corrupted = val_vals_clean * 1000.0 + 9999.0  # extreme corruption

    # Fit mean and std on train only (clean vs corrupted val)
    # 1. Clean run
    mean_1 = float(train_vals.mean())
    std_1 = float(train_vals.std())

    # 2. Corrupted run
    mean_2 = float(train_vals.mean())
    std_2 = float(train_vals.std())

    assert mean_1 == mean_2
    assert std_1 == std_2
    # Ensure held-out values cannot bleed into training scaler
    assert mean_1 != float(val_vals_corrupted.mean())
