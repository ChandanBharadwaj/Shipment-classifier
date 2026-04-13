"""Load and validate taxonomy definitions from JSON configuration."""

import json
from pathlib import Path

from minerva.schema import TaxonomyGroup


def load_taxonomy(path: Path) -> list[TaxonomyGroup]:
    """Parse taxonomy.json into validated TaxonomyGroup objects.

    Args:
        path: Path to the taxonomy JSON file.

    Returns:
        List of validated TaxonomyGroup instances.

    Raises:
        FileNotFoundError: If the taxonomy file does not exist.
        ValueError: If the taxonomy file is malformed or contains invalid data.
    """
    if not path.exists():
        raise FileNotFoundError(f"Taxonomy file not found: {path}")

    with open(path) as f:
        data = json.load(f)

    if "groups" not in data:
        raise ValueError("Taxonomy JSON must contain a 'groups' key")

    groups: list[TaxonomyGroup] = []
    seen_ids: set[str] = set()

    for entry in data["groups"]:
        if "group_id" not in entry:
            raise ValueError(f"Taxonomy group missing 'group_id': {entry}")

        group_id = entry["group_id"]
        if group_id in seen_ids:
            raise ValueError(f"Duplicate taxonomy group_id: {group_id}")
        seen_ids.add(group_id)

        if not entry.get("semantic_phrases"):
            raise ValueError(
                f"Taxonomy group '{group_id}' must have at least one semantic phrase"
            )

        groups.append(TaxonomyGroup(**entry))

    return groups
