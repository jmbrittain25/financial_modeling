"""
Lightweight persistence helpers for ScenarioConfig and DistributionLibrary.

Phase 1: in-memory + JSON string round-tripping (used by tests).
Phase 3+: file I/O for user_data/ and template loading.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import ScenarioConfig, DistributionLibrary, SavedDistribution

# Default location for committed templates (relative to this file or repo root)
TEMPLATES_DIR = Path(__file__).parent.parent.parent.parent / "app" / "data" / "templates"


def scenario_to_json(cfg: ScenarioConfig, indent: int = 2) -> str:
    return cfg.to_json(indent=indent)


def scenario_from_json(text: str) -> ScenarioConfig:
    return ScenarioConfig.from_json(text)


def load_scenario(path: Path) -> ScenarioConfig:
    text = path.read_text(encoding="utf-8")
    return scenario_from_json(text)


def save_scenario(cfg: ScenarioConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(scenario_to_json(cfg), encoding="utf-8")


def load_distribution_library(path: Path) -> DistributionLibrary:
    if not path.exists():
        return DistributionLibrary()
    data = json.loads(path.read_text(encoding="utf-8"))
    return DistributionLibrary.from_dict(data)


def save_distribution_library(lib: DistributionLibrary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lib.to_dict(), indent=2), encoding="utf-8")


# -----------------------------------------------------------------------------
# Template helpers (Phase 3+)
# -----------------------------------------------------------------------------

def list_templates() -> List[str]:
    """Return names of all committed scenario templates (without .json)."""
    if not TEMPLATES_DIR.exists():
        return []
    return sorted([p.stem for p in TEMPLATES_DIR.glob("*.json")])


def load_template(name: str) -> ScenarioConfig:
    """Load a committed template by short name (e.g. 'variable_rate_mortgage')."""
    path = TEMPLATES_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Template not found: {name} (looked in {TEMPLATES_DIR})")
    return load_scenario(path)


__all__ = [
    "scenario_to_json",
    "scenario_from_json",
    "load_scenario",
    "save_scenario",
    "load_distribution_library",
    "save_distribution_library",
    "list_templates",
    "load_template",
    "TEMPLATES_DIR",
]
