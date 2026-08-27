"""
InnoCore AI 智能体控制器 - LangGraph 多智能体编排
负责四大智能体的协同调度、任务编排和执行日志
"""

import asyncio
import json
import logging
import os
import socket
from contextlib import suppress
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from agents.coach import CoachAgent
from agents.hunter import HunterAgent
from agents.miner import MinerAgent
from agents.validator import ValidatorAgent
from core.config import get_config
from core.concurrency import concurrency_limiter
from core.exceptions import AgentException
from core.request_context import get_request_identity
from core.task_queue import TaskQueueBackend, create_task_backend, utc_now

logger = logging.getLogger(__name__)


class TaskType(Enum):
    PAPER_HUNTING = "paper_hunting"
    PAPER_ANALYSIS = "paper_analysis"
    WRITING_ASSISTANCE = "writing_assistance"
    CITATION_VALIDATION = "citation_validation"
    FULL_WORKFLOW = "full_workflow"
    AUTONOMOUS_RESEARCH = "autonomous_research"


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentController:
    """智能体控制器，支持进程内和 Redis 分布式任务调度。"""

    def __init__(self):
        self.config = get_config()

        self.agents = {
            "hunter": HunterAgent(),
            "miner": MinerAgent(),
            "coach": CoachAgent(),
            "validator": ValidatorAgent(),
        }

        self.task_backend: Optional[TaskQueueBackend] = None
        self.semaphore = asyncio.Semaphore(self.config.concurrent_agents)
        self._redis_available = False
        self._initialized = False
        self._shutdown_event = asyncio.Event()
        self._callbacks: Dict[str, Callable] = {}
        self.instance_id = self.config.task_queue.instance_id or (
            f"{socket.gethostname()}:{os.getpid()}:{uuid4().hex[:8]}"
        )

        # 事件回调
        self.event_callbacks: Dict[str, List[Callable]] = {
            "task_started": [],
            "task_completed": [],
            "task_failed": [],
            "agent_status_changed": [],
        }

    async def initialize(self):
        """初始化控制器"""
        if self._initialized:
            return
        logger.info("初始化 Agent Controller...")
        self.task_backend = create_task_backend()
        await self.task_backend.initialize()
        self._redis_available = self.task_backend.name == "redis_stream"
        self._initialized = True
        logger.info(
            "Agent Controller 初始化完成: backend=%s, worker=%s, instance=%s",
            self.task_backend.name,
            self.config.task_queue.worker_enabled,
            self.instance_id,
        )

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self.initialize()

    async def submit_task(
        self, task_type: TaskType, input_data: Dict[str, Any],
        priority: int = 0, callback: Callable = None,
        task_id: Optional[str] = None,
    ) -> str:
        """提交任务到队列"""
        await self._ensure_initialized()
        task_id = task_id or (
            f"task_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}_"
            f"{uuid4().hex[:8]}"
        )
        identity = get_request_identity()
        input_data = dict(input_data)
        # In local development an explicit user_id remains convenient. In
        # authenticated mode the request identity always wins.
        user_id = identity.user_id
        if user_id == "anonymous" and input_data.get("user_id"):
            user_id = str(input_data["user_id"])
        input_data["user_id"] = user_id
        agent_by_type = {
            TaskType.PAPER_HUNTING: "hunter",
            TaskType.PAPER_ANALYSIS: "miner",
            TaskType.WRITING_ASSISTANCE: "coach",
            TaskType.CITATION_VALIDATION: "validator",
            TaskType.FULL_WORKFLOW: "controller",
            TaskType.AUTONOMOUS_RESEARCH: "controller",
        }
        task = {
            "id": task_id,
            "type": task_type.value,
            "agent_type": agent_by_type[task_type],
            "tenant_id": identity.tenant_id,
            "user_id": user_id,
            "input_data": input_data,
            "status": TaskStatus.PENDING.value,
            "priority": int(priority),
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "started_at": None,
            "completed_at": None,
            "result": None,
            "error": None,
            "owner": None,
            "completed_by": None,
            "lease_token": None,
            "lease_until": None,
            "retry_count": 0,
            "max_retries": self.config.task_queue.max_retries,
            "events": [],
        }
        if callback:
            if self.task_backend.name == "redis_stream":
                raise AgentException(
                    "分布式任务不支持进程内 callback，请使用任务状态或事件接口"
                )
            self._callbacks[task_id] = callback
        await self.task_backend.submit(task)
        logger.info(f"任务已提交: {task_id}, 类型: {task_type.value}")
        return task_id

    async def execute_task(self, task_id: str) -> Dict[str, Any]:
        """执行或等待任务，兼容原有同步业务路由。"""
        await self._ensure_initialized()
        task = await self.task_backend.get(task_id)
        if not task:
            raise AgentException(f"任务不存在: {task_id}")
        if not self._owned_by_request(task):
            raise AgentException(f"任务不存在: {task_id}")
        if task["status"] == TaskStatus.COMPLETED.value:
            return task.get("result") or {}
        if task["status"] in {TaskStatus.FAILED.value, TaskStatus.CANCELLED.value}:
            raise AgentException(task.get("error") or f"任务已{task['status']}")

        # Redis 模式由 Worker 独占执行权；HTTP 进程只等待共享状态。
        if self.task_backend.name == "redis_stream":
            return await self.wait_for_task(task_id)

        claimed = await self.task_backend.claim_task(task_id, self.instance_id)
        if claimed:
            task, lease_token = claimed
            try:
                return await self._execute_claimed_task(
                    task, lease_token, self.instance_id
                )
            except AgentException:
                current = await self.task_backend.get(task_id)
                if current and current["status"] == TaskStatus.PENDING.value:
                    return await self.wait_for_task(task_id)
                raise
        return await self.wait_for_task(task_id)

    async def submit_and_wait(
        self, task_type: TaskType, input_data: Dict[str, Any], priority: int = 0
    ) -> Dict[str, Any]:
        task_id = await self.submit_task(task_type, input_data, priority)
        return await self.execute_task(task_id)

    async def wait_for_task(
        self, task_id: str, timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        deadline = asyncio.get_running_loop().time() + (
            timeout or self.config.task_queue.wait_timeout
        )
        poll_seconds = self.config.task_queue.poll_interval_ms / 1000
        while asyncio.get_running_loop().time() < deadline:
            task = await self.task_backend.get(task_id)
            if not task:
                raise AgentException(f"任务不存在: {task_id}")
            if task["status"] == TaskStatus.COMPLETED.value:
                return task.get("result") or {}
            if task["status"] == TaskStatus.FAILED.value:
                raise AgentException(task.get("error") or "任务执行失败")
            if task["status"] == TaskStatus.CANCELLED.value:
                raise AgentException("任务已取消")
            await asyncio.sleep(poll_seconds)
        raise AgentException(f"等待任务超时: {task_id}")

    async def _execute_claimed_task(
        self, task: Dict[str, Any], lease_token: str, worker_id: str
    ) -> Dict[str, Any]:
        task_id = task["id"]
        async with self.semaphore:
            start_time = datetime.now(timezone.utc)
            task.setdefault("agent_results", {})
            await self._trigger_event("task_started", task)
            heartbeat = asyncio.create_task(
                self._heartbeat_task(task_id, lease_token, worker_id)
            )

            exec_id = None
            try:
                from core.database import db_manager
                exec_id = await db_manager.log_agent_execution(
                    agent_name="controller",
                    task_type=task["type"],
                    task_id=task_id,
                    input_summary=json.dumps(task["input_data"], ensure_ascii=False)[:500],
                )
            except Exception:
                pass

            try:
                result = await self._dispatch_task(task)
                duration_ms = int(
                    (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                )
                persisted = await self.task_backend.complete(
                    task_id, worker_id, lease_token, result
                )
                if not persisted:
                    raise AgentException(
                        f"任务 {task_id} 租约已失效或任务已取消，结果未提交"
                    )
                task.update(
                    status=TaskStatus.COMPLETED.value,
                    completed_at=utc_now(),
                    result=result,
                )

                if exec_id:
                    try:
                        from core.database import db_manager
                        await db_manager.update_agent_execution(
                            exec_id, "completed",
                            output_summary=json.dumps(result, ensure_ascii=False)[:500],
                            duration_ms=duration_ms,
                        )
                    except Exception:
                        pass

                await self._trigger_event("task_completed", task)
                callback = self._callbacks.pop(task_id, None)
                if callback:
                    try:
                        callback_result = callback(task)
                        if asyncio.iscoroutine(callback_result):
                            await callback_result
                    except Exception:
                        logger.exception("任务完成回调失败: task_id=%s", task_id)

                return result

            except Exception as e:
                duration_ms = int(
                    (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                )
                current = await self.task_backend.get(task_id)
                if current and current["status"] == "cancelling":
                    await self.task_backend.fail(
                        task_id, worker_id, lease_token, "task cancelled"
                    )
                    current = await self.task_backend.get(task_id)
                if current and current["status"] == TaskStatus.CANCELLED.value:
                    task.update(
                        status=TaskStatus.CANCELLED.value,
                        completed_at=current.get("completed_at"),
                    )
                    if exec_id:
                        try:
                            from core.database import db_manager
                            await db_manager.update_agent_execution(
                                exec_id, "cancelled", duration_ms=duration_ms,
                            )
                        except Exception:
                            pass
                    raise AgentException(f"任务已取消: {task_id}") from e
                outcome = await self.task_backend.fail(
                    task_id, worker_id, lease_token, str(e)
                )
                task["status"] = (
                    TaskStatus.PENDING.value
                    if outcome == "retry"
                    else TaskStatus.FAILED.value
                )
                task["completed_at"] = utc_now() if outcome == "failed" else None
                task["error"] = str(e)

                if exec_id:
                    try:
                        from core.database import db_manager
                        await db_manager.update_agent_execution(
                            exec_id, "failed", duration_ms=duration_ms, error_message=str(e),
                        )
                    except Exception:
                        pass

                await self._trigger_event("task_failed", task)
                logger.error(f"任务执行失败 {task_id}: {str(e)}")
                raise AgentException(f"任务执行失败: {str(e)}")

            finally:
                heartbeat.cancel()
                with suppress(asyncio.CancelledError):
                    await heartbeat

    async def _heartbeat_task(
        self, task_id: str, lease_token: str, worker_id: str
    ) -> None:
        while True:
            await asyncio.sleep(self.config.task_queue.heartbeat_seconds)
            renewed = await self.task_backend.heartbeat(
                task_id, worker_id, lease_token
            )
            if not renewed:
                return

    async def _dispatch_task(self, task: Dict) -> Dict[str, Any]:
        """分发任务到对应 Agent"""
        dispatch_map = {
            TaskType.PAPER_HUNTING: self._execute_paper_hunting,
            TaskType.PAPER_ANALYSIS: self._execute_paper_analysis,
            TaskType.WRITING_ASSISTANCE: self._execute_writing_assistance,
            TaskType.CITATION_VALIDATION: self._execute_citation_validation,
            TaskType.FULL_WORKFLOW: self._execute_full_workflow,
            TaskType.AUTONOMOUS_RESEARCH: self._execute_autonomous_research,
        }
        try:
            task_type = TaskType(task["type"])
        except ValueError as exc:
            raise AgentException(f"不支持的任务类型: {task['type']}") from exc
        handler = dispatch_map.get(task_type)
        if not handler:
            raise AgentException(f"不支持的任务类型: {task['type']}")
        return await handler(task)

    async def _run_agent(
        self, agent_name: str, input_data: Dict[str, Any], task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a singleton agent with isolated state and distributed limits."""
        agent = self.agents[agent_name]
        tenant_id = task.get("tenant_id", "default")
        user_id = task.get("user_id") or input_data.get("user_id") or "anonymous"
        async with concurrency_limiter.slot(
            f"user:{tenant_id}:{user_id}", self.config.task_queue.user_concurrency
        ):
            if agent_name == "miner":
                async with concurrency_limiter.slot(
                    "agent:miner", self.config.task_queue.miner_concurrency
                ):
                    async with agent.execution_scope(f"{task['id']}:{agent_name}"):
                        return await agent.run(input_data)
            async with agent.execution_scope(f"{task['id']}:{agent_name}"):
                return await agent.run(input_data)

    async def _execute_paper_hunting(self, task: Dict) -> Dict[str, Any]:
        result = await self._run_agent("hunter", task["input_data"], task)
        task["agent_results"]["hunter"] = result
        return {
            "task_type": "paper_hunting",
            "papers_found": result.get("papers", []),
            "statistics": {
                "total_found": result.get("total_found", 0),
                "downloaded": result.get("downloaded_papers", 0),
            },
        }

    async def _execute_paper_analysis(self, task: Dict) -> Dict[str, Any]:
        result = await self._run_agent("miner", task["input_data"], task)
        task["agent_results"]["miner"] = result
        return {
            "task_type": "paper_analysis",
            "analysis_report": result,
            "paper_id": task["input_data"].get("paper_id"),
        }

    async def _execute_writing_assistance(self, task: Dict) -> Dict[str, Any]:
        result = await self._run_agent("coach", task["input_data"], task)
        task["agent_results"]["coach"] = result
        return {
            "task_type": "writing_assistance",
            # "assistance_result": result['result']['polished_text'],
            "assistance_result": result,
            "user_id": task["input_data"].get("user_id"),
        }

    async def _execute_citation_validation(self, task: Dict) -> Dict[str, Any]:
        result = await self._run_agent("validator", task["input_data"], task)
        task["agent_results"]["validator"] = result
        return {
            "task_type": "citation_validation",
            "validation_result": result,
            "paper_info": task["input_data"].get("paper_info"),
        }

    async def _execute_autonomous_research(self, task: Dict) -> Dict[str, Any]:
        from agents.autonomous import AutonomousResearchGraph

        input_data = task["input_data"]

        async def progress(kind: str, message: str, data: Dict[str, Any]):
            await self.append_task_event(
                task["id"],
                {
                    "time": utc_now(),
                    "type": kind,
                    "message": message,
                    "data": data,
                },
            )

        try:
            graph = AutonomousResearchGraph(
                self.agents,
                agent_runner=lambda name, payload: self._run_agent(
                    name, payload, task
                ),
            )
            result = await graph.run(
                input_data["message"],
                input_data.get("thread_id") or task["id"],
                input_data.get("context", {}),
                progress,
            )
            return {"answer": result["answer"], "result": result}
        except Exception as exc:
            await progress("failed", f"Execution failed: {exc}", {})
            raise

    async def _execute_full_workflow(self, task: Dict) -> Dict[str, Any]:
        """执行完整工作流: Hunter -> Miner(xN) -> Validator -> Coach"""
        input_data = task["input_data"]
        user_id = input_data.get("user_id")
        keywords = input_data.get("keywords", [])

        workflow_result = {
            "task_type": "full_workflow",
            "stages": {},
            "final_papers": [],
            "analysis_reports": [],
            "warnings": [],
        }

        # 记录工作流
        wf_id = None
        try:
            from core.database import db_manager
            wf_id = await db_manager.create_workflow(
                user_id or "anonymous", "full",
                steps=["hunting", "analysis", "validation", "coach"],
            )
        except Exception:
            pass

        try:
            # Stage 1: Hunter - 论文搜索
            logger.info("工作流 Stage 1: 论文搜索")
            hunting_result = await self._run_agent("hunter", {
                "keywords": keywords,
                "max_papers": input_data.get("max_papers", 10),
                "sources": input_data.get("sources", ["arxiv"]),
            }, task)
            if hunting_result.get("status") in {"failed", "error"}:
                raise AgentException(
                    f"Hunter 阶段失败: {hunting_result.get('error', 'unknown error')}"
                )
            workflow_result["stages"]["hunting"] = hunting_result
            task["agent_results"]["hunter"] = hunting_result
            papers = hunting_result.get("papers", [])

            # Stage 2: Miner - 并行分析
            logger.info(f"[Workflow] Stage 2: 并行分析 {len(papers)} 篇论文")
            analysis_tasks = []
            for paper in papers[:5]:
                miner_input = None
                if paper.get("db_id"):
                    miner_input = {"paper_id": paper["db_id"], "user_id": user_id, "analysis_type": "full"}
                elif paper.get("pdf_url"):
                    miner_input = {"paper_url": paper["pdf_url"], "user_id": user_id, "analysis_type": "full"}
                elif paper.get("title") and paper.get("abstract"):
                    miner_input = {
                        "title": paper["title"], "abstract": paper["abstract"],
                        "authors": paper.get("authors", []), "user_id": user_id, "analysis_type": "full",
                    }
                if miner_input:
                    logger.info(
                        "[Workflow] 提交 MinerAgent 分析: %s",
                        paper.get("title", "")[:60],
                    )
                    analysis_tasks.append(
                        self._run_agent("miner", miner_input, task)
                    )
                else:
                    logger.warning(
                        "[Workflow] 跳过论文（缺少可用的标识符）: %s",
                        paper.get("title", "")[:60],
                    )

            if analysis_tasks:
                analyses = await asyncio.gather(*analysis_tasks, return_exceptions=True)
                for analysis in analyses:
                    if isinstance(analysis, Exception):
                        warning = f"Miner 分析失败: {str(analysis)}"
                        workflow_result["warnings"].append(warning)
                        logger.warning("[Workflow] %s", warning)
                    elif not isinstance(analysis, dict) or analysis.get("success") is not True:
                        error = (
                            analysis.get("error", "success flag is not true")
                            if isinstance(analysis, dict) else "non-object response"
                        )
                        workflow_result["warnings"].append(
                            f"Miner 返回无效分析: {error}"
                        )
                    else:
                        workflow_result["analysis_reports"].append(analysis)
            else:
                logger.warning("[Workflow] Stage 2: 没有论文需要分析（所有论文缺少 db_id/url/title）")

            # Stage 3: Validator - 引用生成
            if input_data.get("validate_citations", False):
                logger.info("工作流 Stage 3: 引用校验")
                for paper in papers:
                    try:
                        v_result = await self._run_agent("validator", {
                            "paper_info": {
                                "title": paper.get("title", ""),
                                "authors": paper.get("authors", []),
                                "doi": paper.get("doi", ""),
                                "year": datetime.now().year,
                            },
                            "formats": ["bibtex", "apa"],
                            "verify_external": True,
                        }, task)
                        if v_result.get("status") in {"failed", "error"}:
                            raise AgentException(
                                v_result.get("error", "Validator returned failure")
                            )
                        paper["citations"] = v_result.get("citations", {})
                    except Exception as e:
                        warning = f"Validator 失败（不阻断工作流）: {str(e)}"
                        workflow_result["warnings"].append(warning)
                        logger.warning(warning)

            # Stage 4: Coach only has a factual basis when at least one Miner
            # report is valid. It is deliberately skipped otherwise.
            writing_task = input_data.get("writing_task")
            if writing_task and workflow_result["analysis_reports"]:
                coach_task_type = (
                    writing_task
                    if writing_task in {"explain", "polish", "mimic", "suggest"}
                    else "suggest"
                )
                coach_result = await self._run_agent("coach", {
                    "user_id": user_id,
                    "task_type": coach_task_type,
                    "content": json.dumps(
                        workflow_result["analysis_reports"],
                        ensure_ascii=False, default=str,
                    ),
                    "instruction": str(writing_task),
                }, task)
                workflow_result["stages"]["coach"] = coach_result
            elif writing_task:
                workflow_result["stages"]["coach"] = {"status": "skipped"}
                workflow_result["warnings"].append(
                    "Coach 已跳过：没有至少一份有效的 Miner 分析"
                )

            workflow_result["stages"]["analysis"] = {
                "policy": "BEST_EFFORT",
                "valid_count": len(workflow_result["analysis_reports"]),
            }

            workflow_result["final_papers"] = papers

            if wf_id:
                try:
                    from core.database import db_manager
                    await db_manager.update_workflow(wf_id, "completed", workflow_result)
                except Exception:
                    pass

            logger.info("完整工作流执行完成")
            return workflow_result

        except Exception as e:
            if wf_id:
                try:
                    from core.database import db_manager
                    await db_manager.update_workflow(wf_id, "failed", {"error": str(e)})
                except Exception:
                    pass
            raise

    async def start_task_processor(self):
        """启动后台任务处理器"""
        await self._ensure_initialized()
        if not self.config.task_queue.worker_enabled:
            logger.info("当前实例未启用任务 Worker")
            return
        logger.info(
            "任务处理器已启动: workers=%s, backend=%s",
            self.config.concurrent_agents,
            self.task_backend.name,
        )
        workers = [
            asyncio.create_task(self._worker_loop(index))
            for index in range(self.config.concurrent_agents)
        ]
        try:
            await asyncio.gather(*workers)
        finally:
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

    async def _worker_loop(self, index: int) -> None:
        worker_id = f"{self.instance_id}:worker-{index}"
        poll_seconds = self.config.task_queue.poll_interval_ms / 1000
        while not self._shutdown_event.is_set():
            try:
                if hasattr(self.task_backend, "heartbeat_worker"):
                    await self.task_backend.heartbeat_worker(worker_id)
                claimed = await self.task_backend.claim_next(worker_id)
                if not claimed:
                    await asyncio.sleep(poll_seconds)
                    continue
                task, lease_token = claimed
                await self._execute_claimed_task_for_worker(
                    task, lease_token, worker_id
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("任务处理器异常: worker=%s, error=%s", worker_id, e)
                await asyncio.sleep(1)

    async def _execute_claimed_task_for_worker(
        self, task: Dict[str, Any], lease_token: str, worker_id: str
    ) -> None:
        try:
            await self._execute_claimed_task(task, lease_token, worker_id)
        except AgentException:
            # Failure state and retries are handled by _execute_claimed_task.
            pass

    async def get_task_status(self, task_id: str) -> Optional[Dict]:
        """获取任务状态"""
        await self._ensure_initialized()
        task = await self.task_backend.get(task_id)
        if not task or not self._owned_by_request(task):
            return None
        return {
            "id": task["id"],
            "type": task["type"],
            "status": task["status"],
            "created_at": task["created_at"],
            "started_at": task.get("started_at"),
            "completed_at": task.get("completed_at"),
            "priority": task["priority"],
            "owner": task.get("owner"),
            "completed_by": task.get("completed_by"),
            "retry_count": task.get("retry_count", 0),
        }

    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        await self._ensure_initialized()
        task = await self.task_backend.get(task_id)
        return task if task and self._owned_by_request(task) else None

    async def list_tasks(self, limit: int = 50) -> List[Dict[str, Any]]:
        await self._ensure_initialized()
        identity = get_request_identity()
        unrestricted_dev = (
            identity.user_id == "anonymous"
            and os.getenv("AUTH_REQUIRED", "false").lower() != "true"
        )
        if unrestricted_dev:
            return await self.task_backend.list(limit)
        return await self.task_backend.list(
            limit, identity.tenant_id, identity.user_id
        )

    async def append_task_event(self, task_id: str, event: Dict[str, Any]) -> None:
        await self._ensure_initialized()
        await self.task_backend.append_event(task_id, event)

    async def cancel_task(self, task_id: str) -> bool:
        await self._ensure_initialized()
        task = await self.task_backend.get(task_id)
        if not task or not self._owned_by_request(task):
            return False
        cancelled = await self.task_backend.cancel(task_id)
        if cancelled:
            logger.info(f"任务已取消: {task_id}")
        return cancelled

    @staticmethod
    def _owned_by_request(task: Dict[str, Any]) -> bool:
        identity = get_request_identity()
        if (
            identity.user_id == "anonymous"
            and os.getenv("AUTH_REQUIRED", "false").lower() != "true"
        ):
            return True
        return (
            task.get("tenant_id", "default") == identity.tenant_id
            and task.get("user_id", "anonymous") == identity.user_id
        )

    async def get_agent_status(self) -> Dict[str, Any]:
        await self._ensure_initialized()
        agent_status = {name: agent.get_status() for name, agent in self.agents.items()}
        tasks = await self.task_backend.list(100)
        return {
            "agents": agent_status,
            "active_tasks": sum(
                task["status"] == TaskStatus.RUNNING.value for task in tasks
            ),
            "queued_tasks": await self.task_backend.queue_size(),
            "completed_tasks": sum(
                task["status"] == TaskStatus.COMPLETED.value for task in tasks
            ),
            "max_concurrent": self.config.concurrent_agents,
            "redis_available": self._redis_available,
            "task_backend": self.task_backend.name,
            "worker_enabled": self.config.task_queue.worker_enabled,
            "instance_id": self.instance_id,
        }

    def add_event_callback(self, event_type: str, callback: Callable):
        if event_type in self.event_callbacks:
            self.event_callbacks[event_type].append(callback)

    async def _trigger_event(self, event_type: str, data: Any):
        for cb in self.event_callbacks.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(data)
                else:
                    cb(data)
            except Exception as e:
                logger.error(f"事件回调失败 {event_type}: {str(e)}")

    async def shutdown(self):
        """关闭控制器"""
        logger.info("关闭 Agent Controller...")
        self._shutdown_event.set()
        if self.task_backend:
            await self.task_backend.close()
        self._initialized = False
        logger.info("Agent Controller 已关闭")


agent_controller = AgentController()
