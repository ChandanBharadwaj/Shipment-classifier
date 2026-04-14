"""Denied-party / watchlist management.

Loads configurable denied party lists from JSON. Each entry supports
a name plus optional aliases, country, and source-list identifier.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DeniedParty:
    """A single denied party entry."""

    name: str
    aliases: list[str] = field(default_factory=list)
    country: str | None = None
    list_name: str = "default"

    @property
    def all_names(self) -> list[str]:
        return [self.name] + self.aliases


def load_denied_parties(path: Path) -> list[DeniedParty]:
    """Load a denied party list from JSON.

    Expected format:
        {
            "list_name": "ofac_sdn",
            "parties": [
                {"name": "Example Corp", "aliases": ["Example Corporation"], "country": "XX"},
                ...
            ]
        }
    Or a list of objects.
    """
    if not path.exists():
        return []

    data = json.loads(path.read_text())

    if isinstance(data, dict):
        list_name = data.get("list_name", "default")
        entries = data.get("parties", [])
    elif isinstance(data, list):
        list_name = "default"
        entries = data
    else:
        raise ValueError(f"Invalid denied party list format: {path}")

    parties: list[DeniedParty] = []
    for entry in entries:
        if "name" not in entry:
            continue
        parties.append(
            DeniedParty(
                name=entry["name"],
                aliases=entry.get("aliases", []),
                country=entry.get("country"),
                list_name=entry.get("list_name", list_name),
            )
        )
    return parties
