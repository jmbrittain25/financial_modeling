"""
FastAPI backend for the financial-simulator platform.

Designed to be clean, well-documented, and friendly to future AI agents.
"""

import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..core import (
    SimulationEngine,
    SimulationResult,
    create_continuous_process,
    create_event_builder,
)

app = FastAPI(
    title="Financial Simulator API",
    description="Programmatic access to the financial simulation engine and Monte Carlo capabilities.",
    version="0.1.0",
)


class RunSimulationRequest(BaseModel):
    config: dict[str, Any]
    seed: int | None = None


class RunSimulationResponse(BaseModel):
    job_id: str
    result: dict[str, Any] | None = None
    status: str = "completed"


@app.post("/simulate", response_model=RunSimulationResponse)
async def run_simulation(req: RunSimulationRequest):
    """Run a single simulation from a config dict."""
    try:
        sim = req.config.get("simulation", req.config)

        start = sim["start"]
        end = sim["end"]

        # Very basic builder for now (will be enhanced in later phases)
        builders = [create_event_builder(b) for b in sim.get("builders", [])]
        procs = []
        for p in sim.get("continuous_processes", []):
            procs.append(create_continuous_process(p))

        engine = SimulationEngine(
            name=sim.get("name", "api-simulation"),
            start=start,
            end=end,
            initial_state=sim.get("initial_state", {}),
            event_builders=builders,
            continuous_processes=procs,
            seed=req.seed,
        )
        engine.run()
        result: SimulationResult = engine.get_result()

        return RunSimulationResponse(
            job_id=str(uuid.uuid4()),
            result=result.model_dump(mode="json"),
            status="completed",
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/health")
async def health():
    return {"status": "ok"}
