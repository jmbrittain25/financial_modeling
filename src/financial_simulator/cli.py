"""Typer CLI for the financial-simulator (new core engine).

Entry point declared in pyproject.toml as:
    simulate = "financial_simulator.cli:main"

Usage examples:
    simulate run --config my-sim.json --seed 42 --verbose
    simulate run -c config.yaml --dry-run
    python -m financial_simulator run --config ...

This v1 intentionally supports only simple/modern config shapes
(the structure under "simulation" or top-level with start/end + builders).
Full legacy config.json support (dists, ${} substitution, derived params)
is planned for a follow-up.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Optional

import typer
import yaml

from .core import (
    AppreciationProcess,
    SimulationEngine,
    SimulationResult,
    create_event_builder,
)

app = typer.Typer(
    name="simulate",
    help="Financial simulation platform CLI (new core engine).",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def _root():
    """Financial simulation platform CLI.

    Use `simulate run --config ...` to execute a simulation with the modern
    SimulationEngine.
    """
    pass


def load_config(path: Path) -> dict[str, Any]:
    """Load simulation config from JSON or YAML (by file extension)."""
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix in {".yaml", ".yml"}:
        return yaml.safe_load(text) or {}
    return json.loads(text)


def _create_continuous_process(d: dict[str, Any]) -> AppreciationProcess:
    """Minimal factory for continuous processes (only Appreciation supported in v1)."""
    if d.get("type") == "Appreciation" or "rate" in d:
        data = {k: v for k, v in d.items() if k != "type"}
        return AppreciationProcess.model_validate(data)
    raise ValueError(f"Unsupported continuous process (only Appreciation in v1): {d}")


def build_engine(cfg: dict[str, Any], seed: Optional[int] = None) -> SimulationEngine:
    """Build a SimulationEngine from a simple config dict.

    Accepts either a top-level engine shape or the common {"simulation": {...}} wrapper.
    Uses the core factories (create_event_builder etc.) which already handle
    most legacy timing/value_gen shapes (e.g. interval_days).
    """
    sim = cfg.get("simulation", cfg)

    # Parse dates (accept ISO strings or already-parsed datetimes)
    start = (
        sim["start"]
        if isinstance(sim.get("start"), dt.datetime)
        else dt.datetime.fromisoformat(str(sim["start"]))
    )
    end = (
        sim["end"]
        if isinstance(sim.get("end"), dt.datetime)
        else dt.datetime.fromisoformat(str(sim["end"]))
    )

    # Build using the core factories (the key reuse point)
    builders = [create_event_builder(b) for b in sim.get("builders", [])]

    procs: list[AppreciationProcess] = []
    for p in sim.get("continuous_processes", []):
        procs.append(_create_continuous_process(p))

    initial_state: dict[str, Any] = sim.get("initial_state", {})
    name: str = sim.get("name", "cli-simulation")

    effective_seed = seed if seed is not None else sim.get("seed")

    return SimulationEngine(
        name=name,
        start=start,
        end=end,
        initial_state=initial_state,
        event_builders=builders,
        continuous_processes=procs,
        seed=effective_seed,
    )


@app.command("run")
def run(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to JSON or YAML simulation config.",
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        "-s",
        help="Override random seed for reproducibility.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Optional path to write the SimulationResult as JSON.",
        writable=True,
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate config and show what would run, then exit without executing.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Print extra details (engine construction, timing, etc.).",
    ),
) -> None:
    """Run one simulation with the SimulationEngine from a config file."""
    if not config.exists():
        typer.secho(f"Error: Config file not found: {config}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    if not config.is_file():
        typer.secho(f"Error: {config} is not a regular file", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    try:
        cfg = load_config(config)
    except Exception as exc:
        typer.secho(f"Failed to load config: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc

    try:
        engine = build_engine(cfg, seed=seed)
    except Exception as exc:
        typer.secho(f"Failed to build SimulationEngine: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc

    if verbose:
        typer.echo(f"Engine: {engine.name}")
        typer.echo(f"Period: {engine.start} → {engine.end}")
        typer.echo(f"Builders: {len(engine.event_builders)}")
        typer.echo(f"Continuous processes: {len(engine.continuous_processes)}")
        typer.echo(f"Seed: {engine.seed}")

    if dry_run:
        typer.secho("Dry run complete (no simulation executed).", fg=typer.colors.YELLOW)
        raise typer.Exit(0)

    typer.secho(f"Running simulation '{engine.name}' ...", fg=typer.colors.BLUE)
    engine.run()
    result: SimulationResult = engine.get_result()

    # Concise but useful summary
    final_cash = result.final_state.get("cumulative_cash")
    final_prop = result.final_state.get("property_value")
    summary = [
        f"Completed: {len(result.events)} events",
        f"Final state keys: {list(result.final_state.keys())}",
    ]
    if final_cash is not None:
        summary.append(f"cumulative_cash: {final_cash:.2f}")
    if final_prop is not None:
        summary.append(f"property_value: {final_prop:.2f}")

    typer.echo("\n".join(summary))

    if output:
        try:
            output.write_text(result.model_dump_json(indent=2))
            typer.secho(f"Result written to {output}", fg=typer.colors.GREEN)
        except Exception as exc:
            typer.secho(f"Failed to write output: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(1) from exc


def main() -> None:
    """Entry point for the `simulate` console script and `python -m financial_simulator`."""
    app()


if __name__ == "__main__":
    main()
