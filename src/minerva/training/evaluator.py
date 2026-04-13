"""Evaluation gate for fine-tuned models.

Enforces minimum precision/recall thresholds before a model
can be promoted to production.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, precision_recall_fscore_support
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

from minerva.logging_utils import get_logger
from minerva.training.trainer import LABEL_NAMES, ScreeningDataset

logger = get_logger("training.evaluator")


@dataclass
class EvaluationGate:
    """Minimum thresholds for model promotion."""

    allow_precision: float = 0.95
    restricted_recall: float = 0.98


@dataclass
class EvaluationResult:
    """Result of model evaluation against the gate."""

    passed: bool
    metrics: dict
    gate: EvaluationGate
    failures: list[str]


def evaluate_model(
    model_path: Path,
    test_data_path: Path,
    gate: EvaluationGate | None = None,
    batch_size: int = 64,
) -> EvaluationResult:
    """Evaluate a fine-tuned model against quality gates.

    Args:
        model_path: Path to saved model directory.
        test_data_path: Path to test.jsonl file.
        gate: Quality thresholds. Uses defaults if None.
        batch_size: Evaluation batch size.

    Returns:
        EvaluationResult with pass/fail, metrics, and failure reasons.
    """
    if gate is None:
        gate = EvaluationGate()

    logger.info("Evaluating model from %s", model_path)
    logger.info("Test data: %s", test_data_path)

    tokenizer = DistilBertTokenizer.from_pretrained(str(model_path))
    model = DistilBertForSequenceClassification.from_pretrained(str(model_path))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = ScreeningDataset(test_data_path, tokenizer)

    all_preds: list[int] = []
    all_labels: list[int] = []

    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        batch_items = [dataset[i] for i in range(start, end)]

        input_ids = torch.stack([item["input_ids"] for item in batch_items]).to(device)
        attention_mask = torch.stack(
            [item["attention_mask"] for item in batch_items]
        ).to(device)
        labels = torch.stack([item["labels"] for item in batch_items])

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        preds = outputs.logits.cpu().numpy().argmax(axis=1)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    # Compute per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=[0, 1, 2], zero_division=0
    )

    metrics = {
        "per_class": {
            LABEL_NAMES[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(LABEL_NAMES))
        },
        "classification_report": classification_report(
            all_labels,
            all_preds,
            target_names=LABEL_NAMES,
            zero_division=0,
        ),
        "total_examples": len(all_labels),
    }

    # Check gates
    failures: list[str] = []

    allow_prec = metrics["per_class"]["allowed"]["precision"]
    if allow_prec < gate.allow_precision:
        failures.append(
            f"Allow precision {allow_prec:.4f} < {gate.allow_precision} threshold"
        )

    restricted_rec = metrics["per_class"]["restricted"]["recall"]
    if restricted_rec < gate.restricted_recall:
        failures.append(
            f"Restricted recall {restricted_rec:.4f} < {gate.restricted_recall} threshold"
        )

    passed = len(failures) == 0

    logger.info("Evaluation %s", "PASSED" if passed else "FAILED")
    for failure in failures:
        logger.warning("Gate failure: %s", failure)

    return EvaluationResult(
        passed=passed,
        metrics=metrics,
        gate=gate,
        failures=failures,
    )
