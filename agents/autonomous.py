"""LLM-driven multi-agent supervisor built with LangGraph.

The supervisor owns no fixed workflow. It receives a natural-language goal and
decides which specialist agents to call, in what order, and when the answer is
complete. Specialist agents remain independently testable modules.
"""

import json
import logging
import asyncio
import time
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
        seen_calls = set()

        async def emit(kind: str, message: str, data: Optional[Dict[str, Any]] = None):
            event = {
                "time": datetime.now(timezone.utc).isoformat(),
                "type": kind,
                "message": message,
                "data": data or {},
            }
            traces.append(event)
            logger.info("[real-agent:%s] %s | %s | data=%s", thread_id, kind, message, json.dumps(data or {}, ensure_ascii=False, default=str)[:1000])
            if on_progress:
                await on_progress(kind, message, data or {})

        def specialist_tool(name: str, description: str) -> StructuredTool:
            async def delegate(instruction: str, context: Dict[str, Any] = None) -> str:
                payload = dict(context or {})
                payload.setdefault("instruction", instruction)
                payload.setdefault("request", instruction)
                # Adapt conversational vocabulary to the stable specialist APIs.
                if name == "hunter":
                    payload.setdefault("sources", ["arxiv"])
                    payload.setdefault("max_papers", 5)
                    payload.setdefault("days_back", 30)
                elif name == "miner":
                    analysis_input = payload.pop("analysis_input", None)
                    if isinstance(analysis_input, dict):
                        payload.update({k: v for k, v in analysis_input.items() if v is not None})
                    if payload.get("external_id") and payload.get("source") == "arxiv" and not payload.get("paper_url"):
                        payload["paper_url"] = f"https://arxiv.org/abs/{payload['external_id']}"
                elif name == "coach":
                    payload.setdefault("user_id", "anonymous")
                    payload.setdefault("content", payload.get("text", instruction))
                    payload.setdefault("task_type", payload.get("task", "polish"))
                elif name == "validator" and "citation" in payload:
                    payload.setdefault("citation_text", payload["citation"])
                call_signature = json.dumps({"agent": name, "payload": payload}, sort_keys=True, ensure_ascii=False, default=str)
                if call_signature in seen_calls:
                    await emit("duplicate_skipped", f"跳过重复的 {name} 调用", {"agent": name, "function": f"{name}.run", "instruction": instruction})
                    return json.dumps({"error": "duplicate_call_skipped", "agent": name, "guidance": "Use previous result or change parameters."}, ensure_ascii=False)
                seen_calls.add(call_signature)
                await emit("decision", f"LLM 决定调用 {name}", {"agent": name, "function": f"{name}.run", "instruction": instruction, "input": self._summarize_payload(payload)})
                await emit("agent_started", f"{name} accepted an assignment", {"agent": name, "function": f"{name}.run", "instruction": instruction})
                started = time.monotonic()

                async def specialist_progress(stage: str, message: str, data: Dict[str, Any]):
                    await emit("agent_progress", message, {"agent": name, "stage": stage, **data})

                agent = self.agents[name]
                token = agent.set_progress_callback(specialist_progress)
                try:
                    result = await asyncio.wait_for(agent.run(payload), timeout=agent.timeout)
                    elapsed = round(time.monotonic() - started, 2)
                    await emit("agent_completed", f"{name} completed its assignment", {"agent": name, "function": f"{name}.run", "elapsed_seconds": elapsed})
                    return json.dumps(result, ensure_ascii=False, default=str)
                except asyncio.TimeoutError:
                    elapsed = round(time.monotonic() - started, 2)
                    message = f"{name} timed out after {agent.timeout}s"
                    await emit("agent_failed", message, {"agent": name, "function": f"{name}.run", "elapsed_seconds": elapsed, "timeout_seconds": agent.timeout})
                    return json.dumps({"error": message, "agent": name}, ensure_ascii=False)
                except Exception as exc:
                    elapsed = round(time.monotonic() - started, 2)
                    await emit("agent_failed", f"{name} failed: {exc}", {"agent": name, "function": f"{name}.run", "elapsed_seconds": elapsed})
                    return json.dumps({"error": str(exc), "agent": name}, ensure_ascii=False)
                finally:
                    agent.reset_progress_callback(token)

            return StructuredTool.from_function(
                coroutine=delegate,
                name=f"delegate_to_{name}",
                description=description,
                args_schema=AgentInput,
            )

        tools = [
            specialist_tool("hunter", "Search papers. ArXiv is always available and the default. IEEE is optional: include it in context.sources only when explicitly requested. context requires keywords (list), and may include max_papers, sources, days_back. Source failures are isolated in source_errors."),
            specialist_tool("miner", "Deeply analyse one paper. Prefer Hunter's analysis_input unchanged. db_id/paper_id may contain only a PostgreSQL UUID. For identifiers like 2606.01899v1 use external_id with source='arxiv', paper_url, or title plus abstract. Never pass an ArXiv/IEEE identifier as db_id."),
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
                "error, change parameters or capability; never repeat an identical failing call. For literature "
                "search use ArXiv unless IEEE was explicitly requested. Ask for clarification only when essential. End with a concise Chinese "
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

    @staticmethod
    def _summarize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Keep decisions inspectable without flooding logs with paper bodies."""
        summary = {}
        for key, value in payload.items():
            if key in {"abstract", "full_text", "content", "text"}:
                summary[key] = f"<{len(str(value))} chars>"
            elif isinstance(value, list) and len(value) > 10:
                summary[key] = f"<{len(value)} items>"
            else:
                summary[key] = value
        return summary
