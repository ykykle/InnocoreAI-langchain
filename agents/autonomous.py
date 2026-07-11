"""LLM-driven multi-agent supervisor built with LangGraph.

The supervisor owns no fixed workflow. It receives a natural-language goal and
decides which specialist agents to call, in what order, and when the answer is
complete. Specialist agents remain independently testable modules.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from core.llm_adapter import get_llm_adapter

logger = logging.getLogger(__name__)


class AgentInput(BaseModel):
    """A flexible, JSON-serialisable assignment for a specialist."""

    instruction: str = Field(description="What this specialist must accomplish")
    context: Dict[str, Any] = Field(default_factory=dict, description="Inputs/results gathered so far")


ProgressCallback = Callable[[str, str, Dict[str, Any]], Awaitable[None]]


class AutonomousResearchGraph:
    """ReAct supervisor that delegates work to modular specialist agents."""

    def __init__(self, agents: Dict[str, Any]):
        self.agents = agents
        self.checkpointer = InMemorySaver()

    async def run(
        self,
        request: str,
        thread_id: str,
        context: Optional[Dict[str, Any]] = None,
        on_progress: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        traces: List[Dict[str, Any]] = []

        async def emit(kind: str, message: str, data: Optional[Dict[str, Any]] = None):
            event = {
                "time": datetime.now(timezone.utc).isoformat(),
                "type": kind,
                "message": message,
                "data": data or {},
            }
            traces.append(event)
            logger.info("[real-agent:%s] %s | %s", thread_id, kind, message)
            if on_progress:
                await on_progress(kind, message, data or {})

        def specialist_tool(name: str, description: str) -> StructuredTool:
            async def delegate(instruction: str, context: Dict[str, Any] = None) -> str:
                await emit("agent_started", f"{name} accepted an assignment", {"agent": name, "instruction": instruction})
                payload = dict(context or {})
                payload.setdefault("instruction", instruction)
                payload.setdefault("request", instruction)
                # Adapt conversational vocabulary to the stable specialist APIs.
                if name == "coach":
                    payload.setdefault("user_id", "anonymous")
                    payload.setdefault("content", payload.get("text", instruction))
                    payload.setdefault("task_type", payload.get("task", "polish"))
                elif name == "validator" and "citation" in payload:
                    payload.setdefault("citation_text", payload["citation"])
                try:
                    result = await self.agents[name].run(payload)
                    await emit("agent_completed", f"{name} completed its assignment", {"agent": name})
                    return json.dumps(result, ensure_ascii=False, default=str)
                except Exception as exc:
                    await emit("agent_failed", f"{name} failed: {exc}", {"agent": name})
                    return json.dumps({"error": str(exc), "agent": name}, ensure_ascii=False)

            return StructuredTool.from_function(
                coroutine=delegate,
                name=f"delegate_to_{name}",
                description=description,
                args_schema=AgentInput,
            )

        tools = [
            specialist_tool("hunter", "Search and collect research papers. context requires keywords (list), and may include max_papers, sources, days_back."),
            specialist_tool("miner", "Read and deeply analyse one paper. context should contain paper_id, paper_url, or title plus abstract."),
            specialist_tool("validator", "Verify metadata and generate citations. context requires paper_info and may contain formats."),
            specialist_tool("coach", "Explain concepts, polish text, or draft research writing. context should contain text, task, and optional style."),
        ]
        model = get_llm_adapter().llm
        graph = create_react_agent(
            model=model,
            tools=tools,
            prompt=(
                "You are InnoCore's autonomous research supervisor. Understand the user's natural-language goal, "
                "then independently choose specialist agents and their order. There is no mandatory pipeline. "
                "Reuse tool results as context for later tools. Never invent tool results. If an agent reports an "
                "error, adapt or explain it. Ask for clarification only when essential. End with a concise Chinese "
                "answer describing what was done and the useful result."
            ),
            checkpointer=self.checkpointer,
        )
        await emit("planning", "LLM supervisor is analysing the request")
        prompt = request
        if context:
            prompt += "\n\nAvailable context:\n" + json.dumps(context, ensure_ascii=False, default=str)
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=prompt)]},
            {"configurable": {"thread_id": thread_id}, "recursion_limit": 30},
        )
        messages = result.get("messages", [])
        answer = next((m.content for m in reversed(messages) if getattr(m, "type", "") == "ai" and m.content), "")
        await emit("completed", "LLM supervisor completed the request")
        return {"answer": answer, "trace": traces, "thread_id": thread_id}
