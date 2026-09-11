"""Chronological experiment splits, window contracts, and event-independent purging."""

from typing import Any

import pandas as pd

SPLIT_INTERVALS = {
    "train": {
        "interval_start": "2018-01-01 00:00:00",
        "interval_end": "2018-10-01 00:00:00",
        "permitted_use": "Model and preprocessing fitting only",
    },
    "val_selection": {
        "interval_start": "2018-10-01 00:00:00",
        "interval_end": "2018-11-01 00:00:00",
        "permitted_use": "Hyperparameter selection, early stopping, ablations",
    },
    "val_calibration": {
        "interval_start": "2018-11-01 00:00:00",
        "interval_end": "2018-12-01 00:00:00",
        "permitted_use": "Fit probability calibration functions (Platt sigmoid, isotonic)",
    },
    "val_alert": {
        "interval_start": "2018-12-01 00:00:00",
        "interval_end": "2019-01-01 00:00:00",
        "permitted_use": "Calibration method selection and alert decision threshold tuning",
    },
    "final_test": {
        "interval_start": "2019-01-01 00:00:00",
        "interval_end": "2020-01-01 00:00:00",
        "permitted_use": "Locked final evaluation only (never consulted during development)",
    },
}


def build_split_configuration(events_df: pd.DataFrame) -> dict[str, Any]:
    """Generate the frozen chronological splits configuration (configs/splits.json)."""
    # Identify boundary-crossing events
    boundary_crossers = []
    for part_name in ["train", "val_selection", "val_calibration", "val_alert"]:
        b_time = pd.Timestamp(SPLIT_INTERVALS[part_name]["interval_end"])
        for _, ev in events_df.iterrows():
            onset = pd.Timestamp(ev["onset"])
            end_ex = pd.Timestamp(ev["end_exclusive"])
            if onset < b_time < end_ex:
                boundary_crossers.append({
                    "event_id": ev["event_id"],
                    "pipe_id": ev["pipe_id"],
                    "area": ev["area"],
                    "boundary": str(b_time),
                    "boundary_name": f"{part_name}->next",
                    "onset": str(onset),
                    "end_exclusive": str(end_ex),
                })

    return {
        "partitions": SPLIT_INTERVALS,
        "window_contract": {
            "history_steps": 288,
            "history_duration_hours": 24.0,
            "history_interval": "[origin - 24h + 5m, origin]",
            "forecast_horizon_steps": 12,
            "forecast_duration_hours": 1.0,
            "forecast_interval": "[origin + 5m, origin + 60m]",
            "cadence_minutes": 5,
        },
        "evaluation_protocols": {
            "operational_rolling": {
                "description": (
                    "Standard continuous monitoring protocol. History before partition start is "
                    "permitted as valid past context. Windows whose forecast target horizon crosses "
                    "the partition end boundary are truncated."
                ),
                "future_target_crossing_allowed": False,
                "past_boundary_context_allowed": True,
            },
            "strict_event_independent": {
                "description": (
                    "Strict event-independent evaluation. Any window in an earlier partition "
                    "whose input history or forecast target overlaps a leak event continuing into "
                    "the subsequent partition is purged to prevent temporal leak contamination."
                ),
                "boundary_crossing_events_count": len(boundary_crossers),
                "purging_rule": "Purge if [history_start, target_end] intersects boundary-crossing event interval.",
            },
        },
        "boundary_crossing_events": boundary_crossers,
    }


def generate_window_manifest(
    index: pd.DatetimeIndex,
    events_df: pd.DataFrame,
    partitions: dict[str, dict[str, str]] | None = None,
    history_steps: int = 288,
    forecast_steps: int = 12,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the exhaustive rolling window manifest for 2018 timestamps.

    For each timestamp origin t:
    - History: [t - (history_steps - 1) * 5m, t]
    - Target: [t + 5m, t + forecast_steps * 5m]
    Enforces boundary containment and event purging.
    """
    if partitions is None:
        partitions = SPLIT_INTERVALS
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("index must be unique and increasing")

    # Prepare event intervals
    event_intervals = []
    for _, ev in events_df.iterrows():
        event_intervals.append({
            "event_id": ev["event_id"],
            "pipe_id": ev["pipe_id"],
            "area": ev["area"],
            "onset": pd.Timestamp(ev["onset"]),
            "end_ex": pd.Timestamp(ev["end_exclusive"]),
            "right_censored": ev["right_censored"],
        })

    # Identify boundary-crossing events (events crossing any partition end)
    boundary_crossers = []
    for p_name in ["train", "val_selection", "val_calibration"]:
        b_time = pd.Timestamp(partitions[p_name]["interval_end"])
        for ev in event_intervals:
            if ev["onset"] < b_time < ev["end_ex"]:
                boundary_crossers.append(ev)

    records = []
    sample_id = 1

    total_by_partition = {k: 0 for k in partitions}
    operational_eligible_by_part = {k: 0 for k in partitions}
    strict_eligible_by_part = {k: 0 for k in partitions}
    purged_target_boundary = {k: 0 for k in partitions}
    purged_event_contamination = {k: 0 for k in partitions}

    step_td = pd.Timedelta(5, unit="m")
    hist_td = pd.Timedelta((history_steps - 1) * 5, unit="m")
    tgt_td = pd.Timedelta(forecast_steps * 5, unit="m")
    expected_step = pd.Timedelta(5, unit="m")
    bad_steps = pd.Series(False, index=range(len(index)))
    if len(index) > 1:
        bad_steps.iloc[1:] = index.to_series().diff().iloc[1:].to_numpy() != expected_step
    bad_step_prefix = bad_steps.astype(int).cumsum().to_numpy()

    for i in range(len(index)):
        origin = index[i]

        # Determine partition for origin t
        part_name = None
        for p_k, p_v in partitions.items():
            p_start = pd.Timestamp(p_v["interval_start"])
            p_end = pd.Timestamp(p_v["interval_end"])
            if p_start <= origin < p_end:
                part_name = p_k
                break

        if part_name is None:
            continue

        total_by_partition[part_name] += 1

        hist_start = origin - hist_td
        hist_end = origin
        tgt_start = origin + step_td
        tgt_end = origin + tgt_td

        # Require every source timestamp used by the history and target, not only endpoints.
        history_first_position = i - history_steps + 1
        target_last_position = i + forecast_steps
        endpoints_available = history_first_position >= 0 and target_last_position < len(index)
        window_grid_complete = False
        if endpoints_available:
            breaks_in_window = (
                bad_step_prefix[target_last_position]
                - bad_step_prefix[history_first_position]
            )
            window_grid_complete = (
                breaks_in_window == 0
                and index[history_first_position] == hist_start
                and index[target_last_position] == tgt_end
            )
        hist_available = history_first_position >= 0 and index[history_first_position] == hist_start
        target_available = target_last_position < len(index) and index[target_last_position] == tgt_end

        # Check if target crosses partition boundary
        p_end = pd.Timestamp(partitions[part_name]["interval_end"])
        # Partitions are half-open: a target exactly at p_end is already outside.
        crosses_boundary = tgt_end >= p_end

        # Active events touching this window
        window_events = []
        for ev in event_intervals:
            if max(hist_start, ev["onset"]) < min(tgt_end, ev["end_ex"]):
                window_events.append(ev["event_id"])

        # Check contamination by boundary-crossing events (strict purging)
        touches_boundary_crosser = False
        for ev in boundary_crossers:
            if max(hist_start, ev["onset"]) < min(tgt_end, ev["end_ex"]):
                touches_boundary_crosser = True
                break

        # Operational eligibility: history available and target strictly inside partition
        eligible_op = window_grid_complete and (not crosses_boundary)
        if crosses_boundary and window_grid_complete:
            purged_target_boundary[part_name] += 1

        # Strict eligibility: operational eligible AND not touching boundary-crossing event
        eligible_strict = eligible_op and (not touches_boundary_crosser)
        if eligible_op and touches_boundary_crosser:
            purged_event_contamination[part_name] += 1

        if eligible_op:
            operational_eligible_by_part[part_name] += 1
        if eligible_strict:
            strict_eligible_by_part[part_name] += 1

        records.append({
            "sample_id": f"W_{sample_id:06d}",
            "partition": part_name,
            "origin_timestamp": str(origin),
            "history_start": str(hist_start),
            "history_end": str(hist_end),
            "target_start": str(tgt_start),
            "target_end": str(tgt_end),
            "active_event_ids": ";".join(window_events) if window_events else "None",
            "history_available": hist_available,
            "target_available": target_available,
            "window_grid_complete": window_grid_complete,
            "targets_cross_boundary": crosses_boundary,
            "touches_boundary_crosser": touches_boundary_crosser,
            "eligible_operational": eligible_op,
            "eligible_strict": eligible_strict,
            "purge_reason": (
                "Incomplete_or_missing_window" if not window_grid_complete
                else "Target_crosses_boundary" if crosses_boundary
                else "Touches_boundary_crossing_event" if touches_boundary_crosser
                else "None"
            ),
        })
        sample_id += 1

    manifest_df = pd.DataFrame(records)

    stats = {
        "total_origins": len(manifest_df),
        "total_by_partition": total_by_partition,
        "operational_eligible_by_partition": operational_eligible_by_part,
        "strict_eligible_by_partition": strict_eligible_by_part,
        "purged_target_crosses_boundary": purged_target_boundary,
        "purged_strict_event_contamination": purged_event_contamination,
    }

    return manifest_df, stats
