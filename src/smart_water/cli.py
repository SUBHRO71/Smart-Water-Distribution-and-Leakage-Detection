"""Command line entry point; run smart-water --help."""

import argparse
import json
from pathlib import Path

from smart_water.baseline import evaluate
from smart_water.data import download, load_scada, prepare


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
    args = parser.parse_args()
    if args.command == "download":
        for path in download(args.output, args.year, args.full):
            print(path)
    elif args.command == "prepare":
        print(json.dumps(prepare(args.raw, args.output, args.year), indent=2))
    else:
        if "SCADA_Flows" not in args.input.name:
            parser.error("baseline expects an original SCADA_Flows CSV in m3/h")
        result = evaluate(load_scada(args.input), args.sensor, args.start, args.period)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
