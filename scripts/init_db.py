#!/usr/bin/env python3
"""Initialize the PostgreSQL feedback store schema."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from minerva.feedback.store import FeedbackStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialize Minerva feedback store database"
    )
    parser.add_argument(
        "--dsn",
        required=True,
        help="PostgreSQL connection string (e.g., postgresql://user:pass@localhost:5432/minerva)",
    )
    args = parser.parse_args()

    print(f"Connecting to database...")
    store = FeedbackStore(args.dsn)

    print("Creating tables...")
    store.init_schema()

    print("Done. Feedback store schema initialized.")
    store.close()


if __name__ == "__main__":
    main()
