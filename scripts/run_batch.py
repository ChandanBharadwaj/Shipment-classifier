#!/usr/bin/env python3
"""CLI entry point for batch shipment screening.

Usage:
    python scripts/run_batch.py --input shipments.json --output decisions.json
    python scripts/run_batch.py --input shipments.csv --output decisions.csv
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

# Ensure src is on the path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from minerva.config import MinervaSettings
from minerva.logging_utils import setup_logging
from minerva.pipeline import ScreeningPipeline
from minerva.schema import Shipment


def load_shipments(path: Path) -> list[Shipment]:
    """Load shipments from JSON or CSV file."""
    suffix = path.suffix.lower()

    if suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return [Shipment(**item) for item in data]
        elif "shipments" in data:
            return [Shipment(**item) for item in data["shipments"]]
        else:
            raise ValueError("JSON must be a list or contain a 'shipments' key")

    elif suffix == ".csv":
        shipments = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                shipments.append(
                    Shipment(
                        id=row.get("id", row.get("shipment_id", "")),
                        description=row.get("description", ""),
                    )
                )
        return shipments

    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .json or .csv")


def write_results(decisions: list, path: Path) -> None:
    """Write screening decisions to JSON or CSV."""
    suffix = path.suffix.lower()

    if suffix == ".json":
        with open(path, "w") as f:
            json.dump(
                [d.model_dump() for d in decisions],
                f,
                indent=2,
                default=str,
            )

    elif suffix == ".csv":
        if not decisions:
            return
        fieldnames = [
            "shipment_id", "action", "reason",
            "taxonomy_hits_count", "classification_label",
            "classification_confidence",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for d in decisions:
                writer.writerow(
                    {
                        "shipment_id": d.shipment_id,
                        "action": d.action.value,
                        "reason": d.reason,
                        "taxonomy_hits_count": len(d.taxonomy_hits),
                        "classification_label": (
                            d.classification.label.value
                            if d.classification
                            else ""
                        ),
                        "classification_confidence": (
                            f"{d.classification.confidence:.4f}"
                            if d.classification
                            else ""
                        ),
                    }
                )
    else:
        raise ValueError(f"Unsupported output format: {suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minerva batch shipment screening"
    )
    parser.add_argument(
        "--input", "-i", required=True, type=Path,
        help="Input file (JSON or CSV) with shipments",
    )
    parser.add_argument(
        "--output", "-o", required=True, type=Path,
        help="Output file (JSON or CSV) for screening decisions",
    )
    parser.add_argument(
        "--config-dir", type=Path, default=None,
        help="Override config directory path",
    )
    args = parser.parse_args()

    settings = MinervaSettings()
    if args.config_dir:
        settings.config_dir = args.config_dir

    setup_logging(settings.log_level)

    print(f"Loading shipments from {args.input}...")
    shipments = load_shipments(args.input)
    print(f"Loaded {len(shipments)} shipments")

    print("Initializing screening pipeline...")
    pipeline = ScreeningPipeline(settings)

    print(f"Screening {len(shipments)} shipments...")
    start_time = time.time()
    decisions = pipeline.screen_batch(shipments)
    elapsed = time.time() - start_time

    # Summary
    from collections import Counter
    summary = Counter(d.action.value for d in decisions)
    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"  Total:         {len(decisions)}")
    for action, count in sorted(summary.items()):
        print(f"  {action:15s} {count}")

    write_results(decisions, args.output)
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
