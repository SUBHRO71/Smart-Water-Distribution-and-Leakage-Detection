"""Command line entry point; run smart-water --help."""

import argparse
import json
from pathlib import Path

import pandas as pd

from smart_water.audit import audit_raw_dataset, generate_coverage_svg, parse_network_topology
from smart_water.baseline import evaluate
from smart_water.data import download, load_scada, prepare
from smart_water.inventory import extract_leak_events
from smart_water.leakage_audit import audit_feature_leakage
from smart_water.leakdb import prepare_hanoi_archive, scenario_fold_feasibility
from smart_water.splits import build_split_configuration, generate_window_manifest
from smart_water.targets import build_targets_config, compute_label_balance, create_target_matrix
from smart_water.validation import expanding_time_folds, fold_feasibility
from smart_water.windows import (
    build_scenario_windows,
    fit_transform_fold,
    leave_one_scenario_out_masks,
    split_development_test,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("download", help="Fetch checksum-verified BattLeDIM files")
    fetch.add_argument("--year", type=int, choices=[2018, 2019], default=2018)
    fetch.add_argument("--full", action="store_true", help="Include other sensors and leak labels")
    fetch.add_argument("--output", type=Path, default=Path("data/raw/battledim"))
    prep = sub.add_parser("prepare", help="Validate, align and normalize available sensor groups")
    prep.add_argument("--year", type=int, choices=[2018, 2019], default=2018)
    prep.add_argument("--raw", type=Path, default=Path("data/raw/battledim"))
    prep.add_argument("--output", type=Path, default=Path("data/processed/battledim"))
    baseline = sub.add_parser("baseline", help="Evaluate rolling flow forecast on a time holdout")
    baseline.add_argument("--input", type=Path,
                          default=Path("data/raw/battledim/2018_SCADA_Flows.csv"))
    baseline.add_argument("--sensor", default="p227")
    baseline.add_argument("--start", default="2018-10-01")
    baseline.add_argument("--period", type=int, default=288)
    baseline.add_argument("--output", type=Path, default=Path("reports/baseline.json"))
    audit_cmd = sub.add_parser("audit", help="Audit raw sensor coverage and network topology")
    audit_cmd.add_argument("--raw", type=Path, default=Path("data/raw/battledim"))
    audit_cmd.add_argument("--inp", type=Path, default=Path("data/raw/battledim/L-TOWN.inp"))
    audit_cmd.add_argument("--output", type=Path, default=Path("reports/audit"))
    inv_cmd = sub.add_parser("inventory", help="Extract pipe-level leak event inventory")
    inv_cmd.add_argument("--leaks", type=Path, default=Path("data/raw/battledim/2018_Leakages.csv"))
    inv_cmd.add_argument("--inp", type=Path, default=Path("data/raw/battledim/L-TOWN.inp"))
    inv_cmd.add_argument("--threshold", type=float, default=0.0)
    inv_cmd.add_argument("--output", type=Path, default=Path("reports/audit"))
    tgt_cmd = sub.add_parser("targets", help="Compute label balance and generate target contract")
    tgt_cmd.add_argument("--leaks", type=Path, default=Path("data/raw/battledim/2018_Leakages.csv"))
    tgt_cmd.add_argument("--events", type=Path, default=Path("reports/audit/leak_events.csv"))
    tgt_cmd.add_argument("--demands", type=Path, default=Path("data/raw/battledim/2018_SCADA_Demands.csv"))
    tgt_cmd.add_argument("--config-output", type=Path, default=Path("configs/targets.json"))
    tgt_cmd.add_argument("--reports-output", type=Path, default=Path("reports/audit/label_balance.csv"))
    splits_cmd = sub.add_parser("splits", help="Freeze chronological splits, purging, and manifests")
    splits_cmd.add_argument("--events", type=Path, default=Path("reports/audit/leak_events.csv"))
    splits_cmd.add_argument("--flows", type=Path, default=Path("data/raw/battledim/2018_SCADA_Flows.csv"))
    splits_cmd.add_argument("--config-output", type=Path, default=Path("configs/splits.json"))
    splits_cmd.add_argument("--manifest-sample", type=Path, default=Path("reports/audit/split_manifest_sample.csv"))
    cv_cmd = sub.add_parser("cv-feasibility", help="Audit ten chronological folds before training")
    cv_cmd.add_argument("--events", type=Path, default=Path("reports/audit/leak_events.csv"))
    cv_cmd.add_argument("--flows", type=Path, default=Path("data/raw/battledim/2018_SCADA_Flows.csv"))
    cv_cmd.add_argument("--output", type=Path, default=Path("reports/audit/cv_feasibility.csv"))
    cv_cmd.add_argument("--folds", type=int, default=10)
    cv_cmd.add_argument("--smote-k", type=int, default=5)
    leak_audit = sub.add_parser("leakage-audit", help="Check selected features before training")
    leak_audit.add_argument(
        "--features", type=Path, default=Path("data/processed/battledim/2018_features.csv")
    )
    leak_audit.add_argument("--events", type=Path, default=Path("reports/audit/leak_events.csv"))
    leak_audit.add_argument("--output", type=Path, default=Path("reports/audit"))
    leakdb_cmd = sub.add_parser("prepare-leakdb", help="Prepare LeakDB Hanoi pressure scenarios")
    leakdb_cmd.add_argument(
        "--archive",
        type=Path,
        default=Path("data/raw/LeakDB-source/CCWI-WDSA2018/Benchmarks/Hanoi_CMH.zip"),
    )
    leakdb_cmd.add_argument(
        "--output", type=Path, default=Path("data/processed/leakdb_hanoi")
    )
    leakdb_cmd.add_argument(
        "--report-output", type=Path, default=Path("reports/audit/leakdb_cv_feasibility.csv")
    )
    window_cmd = sub.add_parser(
        "preprocess-leakdb", help="Build LeakDB windows and verify fold-local scaling/SMOTE"
    )
    window_cmd.add_argument(
        "--input", type=Path,
        default=Path("data/processed/leakdb_hanoi/hanoi_pressure_labels.csv"),
    )
    window_cmd.add_argument("--history", type=int, default=24)
    window_cmd.add_argument("--stride", type=int, default=6)
    window_cmd.add_argument("--smote-k", type=int, default=5)
    window_cmd.add_argument("--seed", type=int, default=42)
    window_cmd.add_argument("--smoke-fold", type=int, choices=range(1, 11), default=1)
    window_cmd.add_argument(
        "--output", type=Path, default=Path("reports/preprocessing/leakdb_windows.json")
    )
    args = parser.parse_args()
    if args.command == "download":
        for path in download(args.output, args.year, args.full):
            print(path)
    elif args.command == "prepare":
        print(json.dumps(prepare(args.raw, args.output, args.year), indent=2))
    elif args.command == "audit":
        args.output.mkdir(parents=True, exist_ok=True)
        audit_df, summary = audit_raw_dataset(args.raw, args.inp)
        csv_path = args.output / "channel_coverage.csv"
        audit_df.to_csv(csv_path, index=False)
        svg_path = args.output / "sensor_coverage_profile.svg"
        generate_coverage_svg(audit_df, svg_path)
        print(json.dumps(summary, indent=2))
        print(f"Audit deliverables written to {csv_path} and {svg_path}")
    elif args.command == "inventory":
        args.output.mkdir(parents=True, exist_ok=True)
        network_info = parse_network_topology(args.inp)
        leaks_df = pd.read_csv(args.leaks, sep=";", decimal=",", index_col="Timestamp")
        leaks_df.index = pd.to_datetime(leaks_df.index)
        events_df, unresolved_df = extract_leak_events(leaks_df, network_info, threshold=args.threshold)
        ev_path = args.output / "leak_events.csv"
        unres_path = args.output / "unresolved_pipe_ids.csv"
        events_df.to_csv(ev_path, index=False)
        unresolved_df.to_csv(unres_path, index=False)
        print(f"Extracted {len(events_df)} leak events, {len(unresolved_df)} unresolved pipes.")
        print(f"Deliverables written to {ev_path} and {unres_path}")
    elif args.command == "targets":
        leaks_df = pd.read_csv(args.leaks, sep=";", decimal=",", index_col="Timestamp")
        leaks_df.index = pd.to_datetime(leaks_df.index)
        events_df = pd.read_csv(args.events)
        demands_df = pd.read_csv(args.demands, sep=";", decimal=",", nrows=1)
        amr_cols = [c for c in demands_df.columns if c != "Timestamp"]

        partitions = {
            "Train": ("2018-01-01", "2018-10-01"),
            "Val_Selection": ("2018-10-01", "2018-11-01"),
            "Val_Calibration": ("2018-11-01", "2018-12-01"),
            "Val_Alert": ("2018-12-01", "2019-01-01"),
        }
        balance_df = compute_label_balance(events_df, leaks_df, partitions)
        args.reports_output.parent.mkdir(parents=True, exist_ok=True)
        balance_df.to_csv(args.reports_output, index=False)

        targets_cfg = build_targets_config(amr_cols)
        args.config_output.parent.mkdir(parents=True, exist_ok=True)
        args.config_output.write_text(json.dumps(targets_cfg, indent=2), encoding="utf-8")
        print(f"Label balance written to {args.reports_output}")
        print(f"Target contract written to {args.config_output}")
    elif args.command == "splits":
        events_df = pd.read_csv(args.events)
        flows_df = pd.read_csv(args.flows, sep=";", decimal=",", usecols=["Timestamp"])
        index = pd.DatetimeIndex(pd.to_datetime(flows_df["Timestamp"]))

        splits_cfg = build_split_configuration(events_df)
        args.config_output.parent.mkdir(parents=True, exist_ok=True)
        args.config_output.write_text(json.dumps(splits_cfg, indent=2), encoding="utf-8")

        manifest_df, stats = generate_window_manifest(index, events_df)
        args.manifest_sample.parent.mkdir(parents=True, exist_ok=True)
        # Take a representative sample: first 50, last 50, and boundary transitions
        boundary_samples = manifest_df[manifest_df["targets_cross_boundary"] | manifest_df["touches_boundary_crosser"]].head(50)
        sample_df = pd.concat([manifest_df.head(50), boundary_samples, manifest_df.tail(50)]).drop_duplicates()
        sample_df.to_csv(args.manifest_sample, index=False)

        print(json.dumps(stats, indent=2))
        print(f"Splits configuration written to {args.config_output}")
        print(f"Sample manifest written to {args.manifest_sample}")
    elif args.command == "cv-feasibility":
        events_df = pd.read_csv(args.events)
        timestamps = pd.read_csv(args.flows, sep=";", usecols=["Timestamp"])
        index = pd.DatetimeIndex(pd.to_datetime(timestamps["Timestamp"]))
        development_index = index[index < pd.Timestamp("2018-10-01")]
        targets = create_target_matrix(development_index, events_df)
        folds = expanding_time_folds(development_index, n_splits=args.folds)
        report = fold_feasibility(
            development_index, targets, events_df, folds, smote_k_neighbors=args.smote_k
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.output, index=False)
        print(report.to_string(index=False))
        print(f"Feasibility report written to {args.output}")
    elif args.command == "leakage-audit":
        features = pd.read_csv(args.features, index_col="timestamp", parse_dates=True)
        events_df = pd.read_csv(args.events)
        targets = create_target_matrix(features.index, events_df)
        allowed = [name for name in features if name.startswith("pressures__")]
        development = features.index < pd.Timestamp("2018-10-01")
        summary, correlations = audit_feature_leakage(
            features.loc[development], targets.loc[development], allowed
        )
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "leakage_audit.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        correlations.to_csv(args.output / "feature_target_correlations.csv", index=False)
        print(json.dumps(summary, indent=2))
    elif args.command == "prepare-leakdb":
        summary = prepare_hanoi_archive(args.archive, args.output)
        frame = pd.read_csv(
            args.output / "hanoi_pressure_labels.csv", parse_dates=["timestamp"]
        )
        folds, feasibility = scenario_fold_feasibility(frame)
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        folds.to_csv(args.report_output, index=False)
        (args.report_output.parent / "leakdb_cv_summary.json").write_text(
            json.dumps(feasibility, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))
        print(json.dumps(feasibility, indent=2))
    elif args.command == "preprocess-leakdb":

        frame = pd.read_csv(args.input, parse_dates=["timestamp"])
        features = [
            name for name in frame if name.startswith("pressure__") and name != "pressure__node_1"
        ]
        windows = build_scenario_windows(
            frame, features, history_steps=args.history, stride=args.stride
        )
        development, test = split_development_test(windows)
        train, assessment = leave_one_scenario_out_masks(windows, args.smoke_fold)
        result = fit_transform_fold(
            windows,
            train,
            assessment,
            apply_smote=True,
            random_state=args.seed,
            k_neighbors=args.smote_k,
        )
        report = {
            "features": len(features),
            "history_steps": args.history,
            "stride_steps": args.stride,
            "total_windows": len(windows.y),
            "development_windows": int(development.sum()),
            "locked_test_windows": int(test.sum()),
            "smote_smoke_fold": args.smoke_fold,
            "seed": args.seed,
            "smote_k_neighbors": args.smote_k,
            "train_counts_before": result["train_counts_before"],
            "train_counts_after": result["train_counts_after"],
            "assessment_counts_unchanged": result["assessment_counts"],
            "assessment_rows_unchanged": int(assessment.sum()) == len(result["y_assessment"]),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    else:
        if "SCADA_Flows" not in args.input.name:
            parser.error("baseline expects an original SCADA_Flows CSV in m3/h")
        result = evaluate(load_scada(args.input), args.sensor, args.start, args.period)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
