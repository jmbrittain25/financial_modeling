"""
Lightweight persistence helpers for ScenarioConfig and DistributionLibrary.

Phase 1: in-memory + JSON string round-tripping (used by tests).
Phase 3+: file I/O for user_data/ and template loading.
"""

from __future__ import annotations

import json
from pathlib import Path

from .models import DistributionLibrary, ScenarioConfig, ScenarioLibrary

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


def list_templates() -> list[str]:
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


# -----------------------------------------------------------------------------
# User data persistence (Phase 4+ full save/load for the interactive builder)
# Location: ~/.financial-simulator/v1/ — survives app restarts, portable.
# -----------------------------------------------------------------------------

USER_DATA_ROOT = Path.home() / ".financial-simulator" / "v1"
USER_SCENARIOS_DIR = USER_DATA_ROOT / "scenarios"
USER_DISTRIBUTIONS_FILE = USER_DATA_ROOT / "distribution_library.json"


def get_user_data_dir() -> Path:
    """Return (and ensure) the root user data directory."""
    USER_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    return USER_DATA_ROOT


def ensure_user_dirs() -> None:
    """Create all user data subdirectories."""
    get_user_data_dir()
    USER_SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)


def _load_json_or_default(path: Path, default_factory):
    if not path.exists():
        return default_factory()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return default_factory.__self__.from_dict(data) if hasattr(default_factory, "__self__") else data
    except Exception:
        return default_factory()


def load_user_distribution_library() -> DistributionLibrary:
    """Load (or create empty) the user's personal distribution library."""
    ensure_user_dirs()
    if not USER_DISTRIBUTIONS_FILE.exists():
        return DistributionLibrary()
    try:
        data = json.loads(USER_DISTRIBUTIONS_FILE.read_text(encoding="utf-8"))
        return DistributionLibrary.from_dict(data)
    except Exception:
        return DistributionLibrary()


def save_user_distribution_library(lib: DistributionLibrary) -> None:
    ensure_user_dirs()
    USER_DISTRIBUTIONS_FILE.write_text(json.dumps(lib.to_dict(), indent=2), encoding="utf-8")


def load_user_scenario_library() -> ScenarioLibrary:
    """Load all user-saved scenarios into a ScenarioLibrary (scans directory)."""
    ensure_user_dirs()
    lib = ScenarioLibrary()
    for p in sorted(USER_SCENARIOS_DIR.glob("*.json")):
        try:
            cfg = load_scenario(p)
            lib.scenarios.append(cfg)
        except Exception:
            continue
    return lib


def save_user_scenario(cfg: ScenarioConfig, overwrite: bool = True) -> Path:
    """Save a scenario as its own JSON file under the user scenarios dir.
    Filename is slugified from the scenario name.
    """
    ensure_user_dirs()
    safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "-" for c in cfg.name).strip("-")[:80] or "untitled"
    path = USER_SCENARIOS_DIR / f"{safe_name}.json"
    if path.exists() and not overwrite:
        # append numeric suffix
        i = 1
        while (USER_SCENARIOS_DIR / f"{safe_name}-{i}.json").exists():
            i += 1
        path = USER_SCENARIOS_DIR / f"{safe_name}-{i}.json"
    save_scenario(cfg, path)
    return path


def list_user_scenarios() -> list[tuple[str, Path]]:
    """Return (name, path) pairs for every .json in the user scenarios dir."""
    ensure_user_dirs()
    out = []
    for p in sorted(USER_SCENARIOS_DIR.glob("*.json")):
        try:
            # peek name without full load for speed
            data = json.loads(p.read_text(encoding="utf-8"))
            name = data.get("name", p.stem)
            out.append((name, p))
        except Exception:
            out.append((p.stem, p))
    return out


def load_user_scenario(name_or_path: str | Path) -> ScenarioConfig:
    """Load by display name (searches) or direct Path."""
    if isinstance(name_or_path, Path):
        return load_scenario(name_or_path)
    for name, path in list_user_scenarios():
        if name == name_or_path:
            return load_scenario(path)
    # fallback: try as filename stem
    candidate = USER_SCENARIOS_DIR / f"{name_or_path}.json"
    if candidate.exists():
        return load_scenario(candidate)
    raise FileNotFoundError(f"User scenario not found: {name_or_path}")


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
    # User persistence (new)
    "get_user_data_dir",
    "ensure_user_dirs",
    "load_user_distribution_library",
    "save_user_distribution_library",
    "load_user_scenario_library",
    "save_user_scenario",
    "list_user_scenarios",
    "load_user_scenario",
    "USER_DATA_ROOT",
    "USER_SCENARIOS_DIR",
    "USER_DISTRIBUTIONS_FILE",
]
