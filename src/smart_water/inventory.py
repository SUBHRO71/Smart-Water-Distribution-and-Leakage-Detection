"""Pipe-level leak event inventory extraction and validation."""

from typing import Any

import numpy as np
import pandas as pd


def extract_leak_events(
    leaks_df: pd.DataFrame,
    network_info: dict[str, Any],
    threshold: float = 0.0,
    source_label: str = "BattLeDIM_2018_Leakages",
    cadence_minutes: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract contiguous positive leak-flow intervals per pipe.

    Returns:
        events_df: DataFrame of extracted leak events with physical properties.
        unresolved_df: DataFrame of pipe IDs that could not be mapped to nominal topology.
    """
    if "Timestamp" in leaks_df.columns:
        leaks_df = leaks_df.set_index("Timestamp")
    if not isinstance(leaks_df.index, pd.DatetimeIndex):
        leaks_df.index = pd.to_datetime(leaks_df.index)

    time_step_hours = cadence_minutes / 60.0
    pipes_info = network_info.get("pipes", {})
    pipe_to_area = network_info.get("pipe_to_area", {})

    events = []
    unresolved = []

    pipe_cols = [c for c in leaks_df.columns if c != "Timestamp"]
    first_ts = leaks_df.index[0]
    last_ts = leaks_df.index[-1]

    event_counter = 1

    for pipe_id in pipe_cols:
        # Check mapping to nominal network
        if pipe_id not in pipes_info:
            unresolved.append({
                "pipe_id": pipe_id,
                "unresolved_reason": "Pipe ID not found in nominal INP [PIPES] section",
                "label_source": source_label,
            })
            continue

        p_meta = pipes_info[pipe_id]
        endpoints = f"({p_meta['node1']}, {p_meta['node2']})"
        area = pipe_to_area.get(pipe_id, "Unknown")

        series = leaks_df[pipe_id]
        is_active = (series > threshold).to_numpy()
        is_unknown = series.isna().to_numpy()

        # Group contiguous runs of active/positive samples
        # A contiguous run starts when active=True and previous!=True
        # and ends when active!=True and previous=True
        # If unknown (NaN) occurs, we do not assume leak stopped, but flag incomplete volume
        padded_active = np.pad(is_active.astype(int), (1, 1), "constant", constant_values=0)
        diff = np.diff(padded_active)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        for st, en in zip(starts, ends):
            onset_ts = leaks_df.index[st]
            last_active_ts = leaks_df.index[en - 1]
            end_exclusive_ts = (
                leaks_df.index[en]
                if en < len(leaks_df)
                else last_active_ts + pd.Timedelta(float(cadence_minutes), unit="m")
            )

            flow_values = series.iloc[st:en].to_numpy(dtype=float)
            # Unknown samples terminate a run, but make the adjacent event boundary
            # uncertain. They must never be interpreted as confirmed no-leak samples.
            unknown_before = bool(st > 0 and is_unknown[st - 1])
            unknown_after = bool(en < len(series) and is_unknown[en])
            has_unknown_samples = bool(is_unknown[st:en].any() or unknown_before or unknown_after)

            duration_hours = float((en - st) * time_step_hours)
            volume_m3 = float(np.nansum(flow_values) * time_step_hours)
            peak_flow = float(np.nanmax(flow_values)) if len(flow_values) > 0 else 0.0
            mean_flow = float(np.nanmean(flow_values)) if len(flow_values) > 0 else 0.0

            left_censored = bool(onset_ts == first_ts and is_active[0])
            right_censored = bool(last_active_ts == last_ts and is_active[-1])

            events.append({
                "event_id": f"EV_{event_counter:03d}",
                "pipe_id": pipe_id,
                "node1": p_meta["node1"],
                "node2": p_meta["node2"],
                "endpoints": endpoints,
                "area": area,
                "onset": str(onset_ts),
                "last_active_sample": str(last_active_ts),
                "end_exclusive": str(end_exclusive_ts),
                "duration_hours": duration_hours,
                "peak_flow_m3_h": round(peak_flow, 4),
                "mean_flow_m3_h": round(mean_flow, 4),
                "volume_m3": round(volume_m3, 4),
                "incomplete_volume": has_unknown_samples,
                "unknown_before": unknown_before,
                "unknown_after": unknown_after,
                "left_censored": left_censored,
                "right_censored": right_censored,
                "activity_threshold_m3_h": threshold,
                "label_source": source_label,
            })
            event_counter += 1

    events_df = pd.DataFrame(events)
    if not events_df.empty:
        # Sort chronologically by onset time
        events_df = events_df.sort_values(by=["onset", "pipe_id"]).reset_index(drop=True)
        # Re-assign sequential IDs in chronological order
        events_df["event_id"] = [f"EV_{i+1:03d}" for i in range(len(events_df))]

    unresolved_df = pd.DataFrame(
        unresolved, columns=["pipe_id", "unresolved_reason", "label_source"]
    )
    return events_df, unresolved_df


def reconstruct_positive_mask(
    events_df: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    pipe_id: str | None = None,
) -> np.ndarray:
    """Reconstruct a boolean mask of active leak samples from the extracted events."""
    mask = np.zeros(len(target_index), dtype=bool)
    filtered = events_df if pipe_id is None else events_df[events_df["pipe_id"] == pipe_id]

    for _, row in filtered.iterrows():
        onset = pd.Timestamp(row["onset"])
        end_ex = pd.Timestamp(row["end_exclusive"])
        ev_mask = (target_index >= onset) & (target_index < end_ex)
        mask |= ev_mask

    return mask
