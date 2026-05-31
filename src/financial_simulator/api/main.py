"""
FastAPI backend for the financial-simulator platform.

Designed to be clean, well-documented, and friendly to future AI agents.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import uuid

from ..core import SimulationEngine, create_event_builder, SimulationResult
from ..core.simulation import AppreciationProcess

app = FastAPI(
    title="Financial Simulator API",
    description="Programmatic access to the financial simulation engine and Monte Carlo capabilities.",
    version="0.1.0",
)


class RunSimulationRequest(BaseModel):
    config: Dict[str, Any]
    seed: Optional[int] = None


class RunSimulationResponse(BaseModel):
    job_id: str
    result: Optional[Dict[str, Any]] = None
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
            if p.get("type") == "Appreciation":
                procs.append(AppreciationProcess.model_validate(p))

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
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}
