"""
Lightweight persistence helpers for ScenarioConfig and DistributionLibrary.

Phase 1: in-memory + JSON string round-tripping (used by tests).
Phase 3+: file I/O for user_data/ and template loading.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .models import DistributionLibrary, ScenarioConfig, ScenarioLibrary

# Repo root (financial_modeling/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Default location for committed templates (relative to this file or repo root)
TEMPLATES_DIR = PROJECT_ROOT / "app" / "data" / "templates"


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
# Location: <project>/user_data/ — local to the repo, gitignored, easy to browse.
# -----------------------------------------------------------------------------

USER_DATA_ROOT = PROJECT_ROOT / "user_data"
USER_SCENARIOS_DIR = USER_DATA_ROOT / "scenarios"
USER_DISTRIBUTIONS_FILE = USER_DATA_ROOT / "distribution_library.json"

_LEGACY_USER_DATA_ROOT = Path.home() / ".financial-simulator" / "v1"
_LEGACY_SCENARIOS_DIR = _LEGACY_USER_DATA_ROOT / "scenarios"
_LEGACY_DISTRIBUTIONS_FILE = _LEGACY_USER_DATA_ROOT / "distribution_library.json"


def _migrate_legacy_user_data_if_needed() -> None:
    """Copy scenarios/distributions from ~/.financial-simulator if the new dir is empty."""
    if USER_DATA_ROOT != PROJECT_ROOT / "user_data":
        return
    ensure_user_dirs()
    new_scenarios = list(USER_SCENARIOS_DIR.glob("*.json"))
    if not new_scenarios and _LEGACY_SCENARIOS_DIR.exists():
        for legacy_path in _LEGACY_SCENARIOS_DIR.glob("*.json"):
            target = USER_SCENARIOS_DIR / legacy_path.name
            if not target.exists():
                shutil.copy2(legacy_path, target)
    if not USER_DISTRIBUTIONS_FILE.exists() and _LEGACY_DISTRIBUTIONS_FILE.exists():
        shutil.copy2(_LEGACY_DISTRIBUTIONS_FILE, USER_DISTRIBUTIONS_FILE)


def get_user_data_dir() -> Path:
    """Return (and ensure) the root user data directory under the project root."""
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
        return (
            default_factory.__self__.from_dict(data)
            if hasattr(default_factory, "__self__")
            else data
        )
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
    _migrate_legacy_user_data_if_needed()
    lib = ScenarioLibrary()
    for p in sorted(USER_SCENARIOS_DIR.glob("*.json")):
        try:
            cfg = load_scenario(p)
            lib.scenarios.append(cfg)
        except Exception:
            continue
    return lib


def _slugify_scenario_filename(name: str) -> str:
    return (
        "".join(c if c.isalnum() or c in ("-", "_") else "-" for c in name).strip("-")[:80]
        or "untitled"
    )


def save_user_scenario(
    cfg: ScenarioConfig,
    overwrite: bool = True,
    *,
    file_name: str | None = None,
) -> Path:
    """Save a scenario as its own JSON file under the user scenarios dir.

    ``file_name`` (optional) sets the on-disk filename stem; defaults to ``cfg.name``.
    When ``overwrite`` is False and the target path exists, a numeric suffix is appended.
    """
    ensure_user_dirs()
    safe_name = _slugify_scenario_filename(file_name or cfg.name)
    path = USER_SCENARIOS_DIR / f"{safe_name}.json"
    if path.exists() and not overwrite:
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


def delete_user_scenario(name: str) -> bool:
    """Delete the on-disk JSON for a user scenario by its display name.

    Returns True if a matching file was removed, False otherwise.
    Safe to call even if the scenario only exists in memory or was already deleted.
    """
    ensure_user_dirs()
    target_path: Path | None = None
    for disp_name, p in list_user_scenarios():
        if disp_name == name:
            target_path = p
            break
    if target_path is None:
        # fallback: try exact stem (covers cases where name changed after save)
        candidate = USER_SCENARIOS_DIR / f"{name}.json"
        if candidate.exists():
            target_path = candidate
    if target_path is not None and target_path.exists():
        try:
            target_path.unlink()
            return True
        except Exception:
            return False
    return False


__all__ = [
    "scenario_to_json",
    "scenario_from_json",
    "load_scenario",
    "save_scenario",
    "load_distribution_library",
    "save_distribution_library",
    "list_templates",
    "load_template",
    "PROJECT_ROOT",
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
    "delete_user_scenario",
    "USER_DATA_ROOT",
    "USER_SCENARIOS_DIR",
    "USER_DISTRIBUTIONS_FILE",
]
