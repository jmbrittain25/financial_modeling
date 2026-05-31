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


__all__ = [
    "scenario_to_json",
    "scenario_from_json",
    "load_scenario",
    "save_scenario",
    "load_distribution_library",
    "save_distribution_library",
]
