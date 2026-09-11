"""Target contract definition and event label coverage analysis."""

from typing import Any

import pandas as pd


def compute_label_balance(
    events_df: pd.DataFrame,
    leaks_df: pd.DataFrame,
    partitions: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    """Compute explicit timestamp prevalence and event coverage across partitions and areas.

    Event coverage completeness is designated as 'Unknown (unsupported)' because no
    independent field repair log exists to establish ground-truth unobserved leak absence.
    """
    if "Timestamp" in leaks_df.columns:
        leaks_df = leaks_df.set_index("Timestamp")
    if not isinstance(leaks_df.index, pd.DatetimeIndex):
        leaks_df.index = pd.to_datetime(leaks_df.index)

    rows = []

    # Map pipes to areas from events_df
    pipe_to_area = dict(zip(events_df["pipe_id"], events_df["area"]))

    scopes = ["Network_Wide", "Area_A", "Area_B", "Area_C"]

    for part_name, (start_str, end_str) in partitions.items():
        start_ts = pd.Timestamp(start_str)
        end_ts = pd.Timestamp(end_str)
        part_mask = (leaks_df.index >= start_ts) & (leaks_df.index < end_ts)
        part_leaks = leaks_df.loc[part_mask]
        n_timestamps = len(part_leaks)

        if n_timestamps == 0:
            continue

        for scope in scopes:
            if scope == "Network_Wide":
                active_mask = (part_leaks > 0).any(axis=1)
                scope_events = events_df[
                    (pd.to_datetime(events_df["onset"]) < end_ts)
                    & (pd.to_datetime(events_df["end_exclusive"]) > start_ts)
                ]
            else:
                scope_pipes = [p for p, a in pipe_to_area.items() if a == scope and p in part_leaks.columns]
                if scope_pipes:
                    active_mask = (part_leaks[scope_pipes] > 0).any(axis=1)
                else:
                    active_mask = pd.Series(False, index=part_leaks.index)
                scope_events = events_df[
                    (events_df["area"] == scope)
                    & (pd.to_datetime(events_df["onset"]) < end_ts)
                    & (pd.to_datetime(events_df["end_exclusive"]) > start_ts)
                ]

            pos_count = int(active_mask.sum())
            prevalence_pct = round(pos_count / n_timestamps * 100.0, 4) if n_timestamps > 0 else 0.0

            # Count events whose onset begins inside this partition
            onset_in_part = scope_events[
                (pd.to_datetime(scope_events["onset"]) >= start_ts)
                & (pd.to_datetime(scope_events["onset"]) < end_ts)
            ]

            rows.append({
                "partition": part_name,
                "interval_start": str(start_ts),
                "interval_end": str(end_ts),
                "target_scope": scope,
                "total_timestamps": n_timestamps,
                "positive_timestamps": pos_count,
                "negative_timestamps": n_timestamps - pos_count,
                "timestamp_prevalence_pct": prevalence_pct,
                "active_events_in_partition": len(scope_events),
                "new_onset_events": len(onset_in_part),
                "event_label_coverage_completeness": "Unknown (No independent reference log; BattLeDIM label provenance only)",
                "censored_events_count": int(scope_events["right_censored"].sum()),
            })

    return pd.DataFrame(rows)


def build_targets_config(amr_columns: list[str]) -> dict[str, Any]:
    """Generate the locked target contract configuration (configs/targets.json)."""
    return {
        "demand_target": {
            "type": "multi_step_regression",
            "description": "Next-hour sequence of aggregate AMR flow rates for Area C.",
            "target_sensors": sorted(amr_columns),
            "sensor_count": len(amr_columns),
            "aggregation": "sum",
            "unit": "m3/h",
            "horizon_steps": 12,
            "horizon_step_minutes": 5,
            "total_horizon_minutes": 60,
            "loss_function": "HuberLoss",
            "observability_notes": (
                "Sum of 82 AMR meters located exclusively in Area C. "
                "Represents observed customer metered consumption, not total network demand."
            ),
        },
        "leak_targets": {
            "type": "multi_label_binary_classification",
            "description": "Area-level concurrent leak presence detection vector (K=3).",
            "dimension_K": 3,
            "activation": "sigmoid",
            "target_vector": ["leak_Area_A", "leak_Area_B", "leak_Area_C"],
            "simultaneous_leaks_permitted": True,
            "loss_function": "BCEWithLogitsLoss",
            "mask_unknown_labels": True,
            "aggregate_network_label_policy": (
                "Network-wide aggregate label is positive on 97.8% of timestamps and is "
                "strictly rejected as a primary classification target to avoid trivially positive models."
            ),
            "areas": {
                "Area_A": {
                    "description": "Main distribution network (657 junctions).",
                    "development_events_count": 11,
                    "sensor_observability": "High (29 pressure sensors, 2 reservoir inlet flow sensors)",
                    "development_support_status": (
                        "Provisional only: October-December validation is single-class positive"
                    ),
                },
                "Area_B": {
                    "description": "Pressure-reduced zone (31 junctions) downstream of PRV-3.",
                    "development_events_count": 1,
                    "sensor_observability": "Low (1 pressure sensor at n226)",
                    "development_support_status": (
                        "Unsupported for robust CV: one event; most assessment blocks are single-class"
                    ),
                },
                "Area_C": {
                    "description": "Elevated tank-fed zone (93 junctions) downstream of PUMP_1.",
                    "development_events_count": 2,
                    "sensor_observability": "High (3 pressure sensors, 1 tank level T1, 1 pump flow PUMP_1, 82 AMRs)",
                    "development_support_status": (
                        "Unsupported for robust CV: chronic p257 makes most periods single-class positive"
                    ),
                },
            },
            "unsupported_policy": (
                "Targets lacking positive development events are marked unsupported; "
                "their predictions must never be claimed as calibrated probabilities."
            ),
        },
    }


def create_target_matrix(
    index: pd.DatetimeIndex,
    events_df: pd.DataFrame,
    unknown_by_area: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Construct area labels, optionally preserving unknown samples as nullable values."""
    target_matrix = pd.DataFrame(0, index=index, columns=["leak_Area_A", "leak_Area_B", "leak_Area_C"])

    for _, row in events_df.iterrows():
        area = row["area"]
        col_name = f"leak_{area}"
        if col_name in target_matrix.columns:
            onset = pd.Timestamp(row["onset"])
            end_ex = pd.Timestamp(row["end_exclusive"])
            mask = (index >= onset) & (index < end_ex)
            target_matrix.loc[mask, col_name] = 1

    if unknown_by_area is not None:
        expected = {"Area_A", "Area_B", "Area_C"}
        if not expected.issubset(unknown_by_area.columns):
            raise ValueError("unknown_by_area must contain Area_A, Area_B and Area_C")
        aligned = unknown_by_area.reindex(index)
        target_matrix = target_matrix.astype("Int8")
        for area in sorted(expected):
            target_matrix.loc[aligned[area].fillna(True).astype(bool), f"leak_{area}"] = pd.NA

    return target_matrix


def build_unknown_by_area(
    leaks_df: pd.DataFrame,
    pipe_to_area: dict[str, str],
) -> pd.DataFrame:
    """Return True when any pipe label in an area is missing at a timestamp."""
    if "Timestamp" in leaks_df.columns:
        leaks_df = leaks_df.set_index("Timestamp")
    if not isinstance(leaks_df.index, pd.DatetimeIndex):
        leaks_df.index = pd.to_datetime(leaks_df.index)
    result = pd.DataFrame(False, index=leaks_df.index, columns=["Area_A", "Area_B", "Area_C"])
    for area in result.columns:
        columns = [name for name in leaks_df if pipe_to_area.get(name) == area]
        if columns:
            result[area] = leaks_df[columns].isna().any(axis=1)
    return result
