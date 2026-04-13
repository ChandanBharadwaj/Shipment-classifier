"""Structured logging configuration for Minerva."""

import json
import logging
import sys
from typing import Any


class JSONFormatter(logging.Formatter):
    """Format log records as JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)
        if record.exc_info and record.exc_info[1]:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)


def setup_logging(level: str = "INFO") -> None:
    """Configure structured JSON logging for Minerva."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())

    root = logging.getLogger("minerva")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.addHandler(handler)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"minerva.{name}")


def log_disagreement(
    logger: logging.Logger,
    shipment_id: str,
    taxonomy_action: str,
    ai_action: str,
    ai_confidence: float,
    description: str,
) -> None:
    """Log a disagreement between taxonomy/keyword and AI decisions."""
    logger.warning(
        "Screening disagreement detected",
        extra={
            "extra_data": {
                "event": "screening_disagreement",
                "shipment_id": shipment_id,
                "taxonomy_action": taxonomy_action,
                "ai_action": ai_action,
                "ai_confidence": ai_confidence,
                "description_preview": description[:200],
            }
        },
    )
