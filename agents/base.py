"""
InnoCore AI 基础智能体类 - 基于 LangGraph ReAct 框架
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from datetime import datetime
from contextvars import ContextVar
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

from core.config import get_config
from core.exceptions import AgentException, TimeoutException
from core.llm_adapter import get_llm_adapter

logger = logging.getLogger(__name__)
_agent_progress_callback: ContextVar = ContextVar("agent_progress_callback", default=None)


class BaseAgent(ABC):
    """基础智能体抽象类 - LangGraph ReAct 实现"""

    def __init__(self, name: str, llm=None,
                 max_steps: int = None, timeout: int = None):
        self.name = name
        self.config = get_config()
        self.llm = llm or get_llm_adapter()

        self.max_steps = max_steps or self.config.agent_max_steps
        self.timeout = timeout or self.config.agent_timeout

        self.tools: List[Tool] = []
        self.tools_dict: Dict[str, Dict] = {}
        self.created_at = datetime.now()
        # Singleton agent definitions are safe to share; execution state is not.
        self._history_var = ContextVar(f"{name}_history", default=None)
        self._state_var = ContextVar(f"{name}_state", default="idle")
        self._active_executions = 0
        self._active_lock = asyncio.Lock()

    def _history(self) -> List[str]:
        history = self._history_var.get()
        if history is None:
            history = []
            self._history_var.set(history)
        return history

    @asynccontextmanager
    async def execution_scope(self, execution_id: str = ""):
        """Isolate mutable history/state for one concurrent invocation."""
        history_token = self._history_var.set([])
        state_token = self._state_var.set("running")
        async with self._active_lock:
            self._active_executions += 1
        try:
            yield execution_id or uuid4().hex
        finally:
            async with self._active_lock:
                self._active_executions -= 1
            self._state_var.reset(state_token)
            self._history_var.reset(history_token)

    @abstractmethod
    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行智能体任务"""
        pass

    def add_tool(self, tool_name: str, tool_func: Callable, description: str = ""):
        """注册 LangChain Tool"""
        tool = Tool(
            name=tool_name,
            description=description or f"Tool: {tool_name}",
            func=tool_func,
        )
        self.tools.append(tool)
        self.tools_dict[tool_name] = {
            "function": tool_func,
            "description": description,
        }

    def get_tools_description(self) -> str:
        if not self.tools:
            return "暂无可用工具"
        return "\n".join(f"- {t.name}: {t.description}" for t in self.tools)

    async def call_tool(self, tool_name: str, tool_input: Any) -> Any:
        """手动调用工具（用于硬编码流程）"""
        if tool_name not in self.tools_dict:
            raise AgentException(f"工具 '{tool_name}' 不存在")

        try:
            tool_func = self.tools_dict[tool_name]["function"]
            if asyncio.iscoroutinefunction(tool_func):
                result = await asyncio.wait_for(tool_func(tool_input), timeout=self.timeout)
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(tool_func, tool_input), timeout=self.timeout
                )
            self._add_to_history(f"Tool {tool_name}: {str(result)[:200]}")
            return result
        except asyncio.TimeoutError:
            raise TimeoutException(f"工具 '{tool_name}' 执行超时")
        except Exception as e:
            raise AgentException(f"工具 '{tool_name}' 执行失败: {str(e)}")

    async def think(self, prompt: str, context: Dict = None) -> str:
        """直接调用 LLM 进行思考"""
        try:
            messages = []
            system_prompt = self._get_system_prompt()
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))

            full_prompt = prompt
            if context:
                ctx_str = json.dumps(context, ensure_ascii=False, indent=2)
                full_prompt = f"上下文信息:\n{ctx_str}\n\n任务:\n{prompt}"

            history = self._history()
            if history:
                history_str = "\n".join(history[-10:])
                full_prompt += f"\n\n历史记录:\n{history_str}"

            messages.append(HumanMessage(content=full_prompt))

            response = await asyncio.wait_for(
                self.llm.ainvoke(messages), timeout=self.timeout
            )
            response_text = response if isinstance(response, str) else str(response)
            self._add_to_history(f"LLM: {response_text[:200]}")
            return response_text
        except asyncio.TimeoutError:
            raise TimeoutException("LLM思考超时")
        except Exception as e:
            raise AgentException(f"LLM思考失败: {str(e)}")

    def _get_system_prompt(self) -> str:
        return f"你是{self.name}智能体，请根据用户需求完成任务。"

    def _build_agent_graph(self, system_prompt: str = None):
        """构建 LangGraph ReAct Agent"""
        if not self.tools:
            logger.warning(f"Agent {self.name} 没有注册工具，无法创建 ReAct Agent")
            return None

        try:
            llm = self.llm.llm if hasattr(self.llm, 'llm') else self.llm
            prompt = system_prompt or self._get_system_prompt()

            agent_graph = create_react_agent(
                model=llm,
                tools=self.tools,
                prompt=prompt,
                checkpointer=InMemorySaver(),
            )
            logger.info(f"Agent {self.name} LangGraph ReAct Agent 构建成功 (工具数: {len(self.tools)})")
            return agent_graph
        except Exception as e:
            logger.error(f"构建 Agent 失败: {str(e)}")
            return None

    async def run_with_tools(self, input_text: str, thread_id: str = None) -> str:
        """使用 LangGraph ReAct Agent 执行任务，LLM 自主决策工具调用"""
        agent_graph = self._build_agent_graph()
        if not agent_graph:
            return await self.think(input_text)

        try:
            config = {"configurable": {"thread_id": thread_id or uuid4().hex}}
            messages = [HumanMessage(content=input_text)]

            result = await asyncio.wait_for(
                asyncio.to_thread(agent_graph.invoke, {"messages": messages}, config),
                timeout=self.timeout,
            )

            output_messages = result.get("messages", [])
            for msg in reversed(output_messages):
                content = getattr(msg, 'content', '')
                msg_type = getattr(msg, 'type', '')
                if content and msg_type in ('ai', 'AIMessage'):
                    return content

            return str(result)
        except asyncio.TimeoutError:
            raise TimeoutException("Agent执行超时")
        except Exception as e:
            logger.error(f"Agent ReAct 执行失败: {str(e)}")
            return await self.think(input_text)

    def _add_to_history(self, message: str):
        timestamp = datetime.now().isoformat()
        history = self._history()
        history.append(f"[{timestamp}] {message}")
        if len(history) > 100:
            del history[:-50]

    def get_history(self, limit: int = 10) -> List[str]:
        return self._history()[-limit:]

    def clear_history(self):
        self._history_var.set([])

    def set_state(self, state: str):
        self._state_var.set(state)
        logger.info(f"Agent {self.name} state: {state}")

    async def emit_progress(self, stage: str, message: str, **data):
        """Emit optional fine-grained progress to the supervising graph."""
        logger.info("Agent %s progress [%s]: %s", self.name, stage, message)
        callback = _agent_progress_callback.get()
        if callback:
            await callback(stage, message, data)

    def set_progress_callback(self, callback):
        return _agent_progress_callback.set(callback)

    def reset_progress_callback(self, token):
        _agent_progress_callback.reset(token)

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": "running" if self._active_executions else self._state_var.get(),
            "created_at": self.created_at.isoformat(),
            "history_count": len(self._history()),
            "active_executions": self._active_executions,
            "tools_count": len(self.tools),
            "max_steps": self.max_steps,
            "timeout": self.timeout,
        }

    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        required_fields = self.get_required_fields()
        for field in required_fields:
            if field not in input_data:
                raise AgentException(f"缺少必需字段: {field}")
        return True

    @abstractmethod
    def get_required_fields(self) -> List[str]:
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', state='{self._state_var.get()}')"

    def __repr__(self) -> str:
        return self.__str__()
