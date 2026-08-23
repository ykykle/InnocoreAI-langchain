"""Natural-language API for the autonomous LangGraph supervisor."""

from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from agents.controller import TaskType, agent_controller

router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20000)
    thread_id: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)


def _public(task: Dict[str, Any]) -> Dict[str, Any]:
    input_data = task.get("input_data") or {}
    task_result = task.get("result") or {}
    status = "queued" if task["status"] == "pending" else task["status"]
    return {
        "id": task["id"],
        "thread_id": input_data.get("thread_id") or task["id"],
        "status": status,
        "request": input_data.get("message", ""),
        "answer": task_result.get("answer"),
        "result": task_result.get("result"),
        "error": task.get("error"),
        "events": task.get("events", []),
        "created_at": task["created_at"],
        "updated_at": task.get("updated_at") or task["created_at"],
    }


@router.post("/runs", status_code=202)
async def create_run(request: ChatRequest):
    run_id = f"agent_{uuid4().hex[:12]}"
    await agent_controller.submit_task(
        TaskType.AUTONOMOUS_RESEARCH,
        {
            "message": request.message,
            "thread_id": request.thread_id or run_id,
            "context": request.context,
        },
        task_id=run_id,
    )
    task = await agent_controller.get_task(run_id)
    return _public(task)


@router.get("/runs/{run_id}")
async def get_run(run_id: str):
    task = await agent_controller.get_task(run_id)
    if not task or task["type"] != TaskType.AUTONOMOUS_RESEARCH.value:
        raise HTTPException(status_code=404, detail="Run not found")
    return _public(task)


@router.get("/runs")
async def list_runs(limit: int = 20):
    tasks = await agent_controller.list_tasks(limit=min(max(limit * 2, 20), 200))
    runs = [
        _public(task)
        for task in tasks
        if task["type"] == TaskType.AUTONOMOUS_RESEARCH.value
    ]
    return runs[:min(limit, 100)]
