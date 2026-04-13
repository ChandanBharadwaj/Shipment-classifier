"""Export labeled data from the feedback store for model training."""

import json
import random
from pathlib import Path
from typing import Any

from minerva.feedback.store import FeedbackStore
from minerva.logging_utils import get_logger

logger = get_logger("training.exporter")

# Maps reviewer decisions to numeric labels for training
LABEL_MAP = {
    "approve": 0,   # Allowed
    "block": 1,     # Restricted
    "manual_review": 2,  # Needs Review
}


def export_training_data(
    store: FeedbackStore,
    output_dir: Path,
    test_split: float = 0.1,
    val_split: float = 0.1,
    seed: int = 42,
) -> dict[str, int]:
    """Export labeled data from the feedback store as train/val/test JSONL files.

    Args:
        store: Feedback store to query.
        output_dir: Directory to write JSONL files.
        test_split: Fraction of data for test set.
        val_split: Fraction of data for validation set.
        seed: Random seed for reproducible splits.

    Returns:
        Dict with counts per split: {"train": N, "val": N, "test": N}
    """
    records = store.export_labeled_data()

    if not records:
        logger.warning("No labeled data found in feedback store")
        return {"train": 0, "val": 0, "test": 0}

    # Convert to training format
    examples: list[dict[str, Any]] = []
    skipped = 0
    for record in records:
        decision = record.get("final_decision", "").lower()
        if decision not in LABEL_MAP:
            skipped += 1
            continue

        examples.append(
            {
                "text": record["description"],
                "label": LABEL_MAP[decision],
                "label_name": decision,
                "shipment_id": record["shipment_id"],
            }
        )

    if skipped:
        logger.info("Skipped %d records with unmapped labels", skipped)

    # Shuffle and split
    rng = random.Random(seed)
    rng.shuffle(examples)

    n = len(examples)
    n_test = int(n * test_split)
    n_val = int(n * val_split)

    test_data = examples[:n_test]
    val_data = examples[n_test : n_test + n_val]
    train_data = examples[n_test + n_val :]

    # Write JSONL files
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = {"train": train_data, "val": val_data, "test": test_data}

    for split_name, data in splits.items():
        path = output_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for example in data:
                f.write(json.dumps(example) + "\n")
        logger.info("Wrote %d examples to %s", len(data), path)

    return {name: len(data) for name, data in splits.items()}
