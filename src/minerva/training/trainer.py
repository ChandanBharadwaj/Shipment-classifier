"""Fine-tune DistilBERT on labeled screening data."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    Trainer,
    TrainingArguments,
)

from minerva.logging_utils import get_logger

logger = get_logger("training.trainer")

NUM_LABELS = 3
LABEL_NAMES = ["allowed", "restricted", "needs_review"]


@dataclass
class TrainingConfig:
    """Hyperparameters for fine-tuning DistilBERT."""

    base_model: str = "distilbert-base-uncased"
    output_dir: str = "./models/distilbert-finetuned"
    num_epochs: int = 5
    learning_rate: float = 2e-5
    train_batch_size: int = 32
    eval_batch_size: int = 64
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_length: int = 512
    seed: int = 42
    save_total_limit: int = 2
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"


class ScreeningDataset(Dataset):
    """PyTorch dataset for screening classification data."""

    def __init__(
        self,
        path: Path,
        tokenizer: DistilBertTokenizer,
        max_length: int = 512,
    ) -> None:
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._examples: list[dict] = []

        with open(path) as f:
            for line in f:
                self._examples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict:
        example = self._examples[idx]
        encoding = self._tokenizer(
            example["text"],
            truncation=True,
            max_length=self._max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(example["label"], dtype=torch.long),
        }


def train_model(
    data_dir: Path,
    config: TrainingConfig | None = None,
) -> Path:
    """Fine-tune DistilBERT on labeled screening data.

    Args:
        data_dir: Directory containing train.jsonl, val.jsonl files.
        config: Training hyperparameters. Uses defaults if None.

    Returns:
        Path to the saved model directory.
    """
    if config is None:
        config = TrainingConfig()

    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")

    logger.info("Loading tokenizer: %s", config.base_model)
    tokenizer = DistilBertTokenizer.from_pretrained(config.base_model)

    logger.info("Loading training data from %s", train_path)
    train_dataset = ScreeningDataset(train_path, tokenizer, config.max_length)
    logger.info("Training examples: %d", len(train_dataset))

    val_dataset = None
    if val_path.exists():
        val_dataset = ScreeningDataset(val_path, tokenizer, config.max_length)
        logger.info("Validation examples: %d", len(val_dataset))

    logger.info("Loading model: %s", config.base_model)
    model = DistilBertForSequenceClassification.from_pretrained(
        config.base_model,
        num_labels=NUM_LABELS,
    )

    # Set label names for the model
    model.config.id2label = {i: name for i, name in enumerate(LABEL_NAMES)}
    model.config.label2id = {name: i for i, name in enumerate(LABEL_NAMES)}

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        eval_strategy=config.eval_strategy if val_dataset else "no",
        save_strategy=config.save_strategy,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end if val_dataset else False,
        metric_for_best_model=config.metric_for_best_model,
        seed=config.seed,
        logging_dir=f"{config.output_dir}/logs",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    logger.info("Starting training...")
    trainer.train()

    # Save final model and tokenizer
    output_path = Path(config.output_dir)
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    logger.info("Model saved to %s", output_path)

    return output_path
