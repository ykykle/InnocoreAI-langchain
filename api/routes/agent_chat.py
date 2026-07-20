"""Natural-language API for the autonomous LangGraph supervisor."""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from agents.autonomous import AutonomousResearchGraph
from agents.controller import agent_controller

logger = logging.getLogger(__name__)
router = APIRouter()
_runs: Dict[str, Dict[str, Any]] = {}


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20000)
    thread_id: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)


def _public(run: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in run.items() if k != "worker"}


async def _execute(run_id: str, request: ChatRequest):
    run = _runs[run_id]

    async def progress(kind: str, message: str, data: Dict[str, Any]):
        run["events"].append({"time": datetime.now(timezone.utc).isoformat(), "type": kind, "message": message, "data": data})
        run["updated_at"] = datetime.now(timezone.utc).isoformat()

    try:
        run["status"] = "running"
        graph = AutonomousResearchGraph(agent_controller.agents)
        result = await graph.run(request.message, request.thread_id or run_id, request.context, progress)
        run.update(status="completed", answer=result["answer"], result=result)
    except Exception as exc:
        logger.exception("Autonomous run %s failed", run_id)
        run.update(status="failed", error=str(exc))
        await progress("failed", f"Execution failed: {exc}", {})
    finally:
        run["updated_at"] = datetime.now(timezone.utc).isoformat()


@router.post("/runs", status_code=202)
async def create_run(request: ChatRequest):
    run_id = f"agent_{uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).isoformat()
    _runs[run_id] = {"id": run_id, "thread_id": request.thread_id or run_id, "status": "queued", "request": request.message, "answer": None, "error": None, "events": [], "created_at": now, "updated_at": now}
    _runs[run_id]["worker"] = asyncio.create_task(_execute(run_id, request))
    return _public(_runs[run_id])


@router.get("/runs/{run_id}")
async def get_run(run_id: str):
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail="Run not found")
    return _public(_runs[run_id])


@router.get("/runs")
async def list_runs(limit: int = 20):
    return [_public(run) for run in list(reversed(_runs.values()))[:min(limit, 100)]]

