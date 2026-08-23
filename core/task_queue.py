"""Task queue backends for local and distributed execution."""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from .config import get_config
from .redis_manager import redis_manager

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskQueueBackend(ABC):
    name = "unknown"

    @abstractmethod
    async def initialize(self) -> None:
        pass

    @abstractmethod
    async def submit(self, task: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    async def claim_next(self, worker_id: str) -> Optional[Tuple[Dict[str, Any], str]]:
        pass

    @abstractmethod
    async def claim_task(
        self, task_id: str, worker_id: str
    ) -> Optional[Tuple[Dict[str, Any], str]]:
        pass

    @abstractmethod
    async def heartbeat(self, task_id: str, worker_id: str, lease_token: str) -> bool:
        pass

    @abstractmethod
    async def complete(
        self, task_id: str, worker_id: str, lease_token: str, result: Dict[str, Any]
    ) -> bool:
        pass

    @abstractmethod
    async def fail(
        self, task_id: str, worker_id: str, lease_token: str, error: str
    ) -> str:
        """Return either ``retry`` or ``failed``."""

    @abstractmethod
    async def cancel(self, task_id: str) -> bool:
        pass

    @abstractmethod
    async def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    async def list(self, limit: int = 50) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def append_event(self, task_id: str, event: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    async def queue_size(self) -> int:
        pass

    @abstractmethod
    async def close(self) -> None:
        pass


class MemoryTaskQueueBackend(TaskQueueBackend):
    """Process-local backend used by the default single-instance mode."""

    name = "local"

    def __init__(self) -> None:
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._lock = asyncio.Lock()
        self._sequence = 0
        self._history_limit = get_config().task_queue.history_maxlen

    async def initialize(self) -> None:
        return

    async def submit(self, task: Dict[str, Any]) -> None:
        async with self._lock:
            if task["id"] in self._tasks:
                raise ValueError(f"Task already exists: {task['id']}")
            self._tasks[task["id"]] = deepcopy(task)
            self._sequence += 1
            await self._queue.put((-int(task["priority"]), self._sequence, task["id"]))

    def _claim_locked(
        self, task_id: str, worker_id: str
    ) -> Optional[Tuple[Dict[str, Any], str]]:
        task = self._tasks.get(task_id)
        if not task or task["status"] != "pending":
            return None
        lease_token = uuid4().hex
        task.update(
            status="running",
            owner=worker_id,
            lease_token=lease_token,
            lease_until=time.time() + get_config().task_queue.lease_seconds,
            started_at=task.get("started_at") or utc_now(),
            updated_at=utc_now(),
        )
        return deepcopy(task), lease_token

    async def claim_next(self, worker_id: str) -> Optional[Tuple[Dict[str, Any], str]]:
        poll_seconds = get_config().task_queue.poll_interval_ms / 1000
        while True:
            try:
                _, _, task_id = await asyncio.wait_for(
                    self._queue.get(), timeout=poll_seconds
                )
            except asyncio.TimeoutError:
                return None
            async with self._lock:
                claimed = self._claim_locked(task_id, worker_id)
            if claimed:
                return claimed

    async def claim_task(
        self, task_id: str, worker_id: str
    ) -> Optional[Tuple[Dict[str, Any], str]]:
        async with self._lock:
            return self._claim_locked(task_id, worker_id)

    async def heartbeat(self, task_id: str, worker_id: str, lease_token: str) -> bool:
        async with self._lock:
            task = self._tasks.get(task_id)
            if (
                not task
                or task["status"] != "running"
                or task.get("owner") != worker_id
                or task.get("lease_token") != lease_token
            ):
                return False
            task["lease_until"] = time.time() + get_config().task_queue.lease_seconds
            task["updated_at"] = utc_now()
            return True

    async def complete(
        self, task_id: str, worker_id: str, lease_token: str, result: Dict[str, Any]
    ) -> bool:
        async with self._lock:
            task = self._tasks.get(task_id)
            if (
                not task
                or task["status"] != "running"
                or task.get("owner") != worker_id
                or task.get("lease_token") != lease_token
            ):
                return False
            task.update(
                status="completed",
                result=deepcopy(result),
                completed_at=utc_now(),
                updated_at=utc_now(),
                completed_by=worker_id,
                owner=None,
                lease_token=None,
                lease_until=None,
            )
            self._trim_history_locked()
            return True

    async def fail(
        self, task_id: str, worker_id: str, lease_token: str, error: str
    ) -> str:
        async with self._lock:
            task = self._tasks.get(task_id)
            if (
                not task
                or task["status"] != "running"
                or task.get("owner") != worker_id
                or task.get("lease_token") != lease_token
            ):
                return "failed"
            task["retry_count"] += 1
            task["error"] = error
            task["updated_at"] = utc_now()
            task["owner"] = None
            task["lease_token"] = None
            task["lease_until"] = None
            if task["retry_count"] <= task["max_retries"]:
                task["status"] = "pending"
                self._sequence += 1
                await self._queue.put(
                    (-int(task["priority"]), self._sequence, task["id"])
                )
                return "retry"
            task["status"] = "failed"
            task["completed_at"] = utc_now()
            self._trim_history_locked()
            return "failed"

    async def cancel(self, task_id: str) -> bool:
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task or task["status"] in TERMINAL_STATUSES:
                return False
            task.update(
                status="cancelled",
                completed_at=utc_now(),
                updated_at=utc_now(),
                owner=None,
                lease_token=None,
                lease_until=None,
            )
            self._trim_history_locked()
            return True

    async def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            task = self._tasks.get(task_id)
            return deepcopy(task) if task else None

    async def list(self, limit: int = 50) -> List[Dict[str, Any]]:
        async with self._lock:
            tasks = sorted(
                self._tasks.values(), key=lambda item: item["created_at"], reverse=True
            )
            return deepcopy(tasks[:limit])

    async def append_event(self, task_id: str, event: Dict[str, Any]) -> None:
        async with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.setdefault("events", []).append(deepcopy(event))
                task["events"] = task["events"][-self._history_limit :]
                task["updated_at"] = utc_now()

    async def queue_size(self) -> int:
        async with self._lock:
            return sum(
                1 for task in self._tasks.values() if task["status"] == "pending"
            )

    async def close(self) -> None:
        return

    def _trim_history_locked(self) -> None:
        terminal = sorted(
            (
                task
                for task in self._tasks.values()
                if task["status"] in TERMINAL_STATUSES
            ),
            key=lambda item: item.get("completed_at") or "",
            reverse=True,
        )
        for task in terminal[self._history_limit :]:
            self._tasks.pop(task["id"], None)


class RedisTaskQueueBackend(TaskQueueBackend):
    """Redis-backed priority queue with leases and fencing tokens."""

    name = "redis"

    _SUBMIT_SCRIPT = """
if redis.call('EXISTS', KEYS[1]) == 1 then return 0 end
local fields = cjson.decode(ARGV[4])
for field, value in pairs(fields) do
    redis.call('HSET', KEYS[1], field, value)
end
redis.call('ZADD', KEYS[2], ARGV[1], ARGV[2])
redis.call('ZADD', KEYS[3], ARGV[3], ARGV[2])
return 1
"""

    _CLAIM_NEXT_SCRIPT = """
local ids = redis.call('ZRANGE', KEYS[1], 0, 9)
for _, id in ipairs(ids) do
    if redis.call('ZREM', KEYS[1], id) == 1 then
        local task_key = ARGV[1] .. id
        if redis.call('EXISTS', task_key) == 1
           and redis.call('HGET', task_key, 'status') == 'pending' then
            redis.call('HSET', task_key,
                'status', 'running',
                'owner', ARGV[2],
                'lease_token', ARGV[3],
                'lease_until', ARGV[4],
                'updated_at', ARGV[5])
            if redis.call('HGET', task_key, 'started_at') == '' then
                redis.call('HSET', task_key, 'started_at', ARGV[5])
            end
            redis.call('ZADD', KEYS[2], ARGV[4], id)
            return id
        end
    end
end
return nil
"""

    _CLAIM_TASK_SCRIPT = """
if redis.call('ZREM', KEYS[1], ARGV[1]) ~= 1 then return nil end
local task_key = ARGV[2] .. ARGV[1]
if redis.call('EXISTS', task_key) ~= 1
   or redis.call('HGET', task_key, 'status') ~= 'pending' then
    return nil
end
redis.call('HSET', task_key,
    'status', 'running',
    'owner', ARGV[3],
    'lease_token', ARGV[4],
    'lease_until', ARGV[5],
    'updated_at', ARGV[6])
if redis.call('HGET', task_key, 'started_at') == '' then
    redis.call('HSET', task_key, 'started_at', ARGV[6])
end
redis.call('ZADD', KEYS[2], ARGV[5], ARGV[1])
return ARGV[1]
"""

    _HEARTBEAT_SCRIPT = """
if redis.call('HGET', KEYS[1], 'status') ~= 'running'
   or redis.call('HGET', KEYS[1], 'owner') ~= ARGV[1]
   or redis.call('HGET', KEYS[1], 'lease_token') ~= ARGV[2] then
    return 0
end
redis.call('HSET', KEYS[1], 'lease_until', ARGV[3], 'updated_at', ARGV[4])
redis.call('ZADD', KEYS[2], ARGV[3], ARGV[5])
return 1
"""

    _COMPLETE_SCRIPT = """
if redis.call('HGET', KEYS[1], 'status') ~= 'running'
   or redis.call('HGET', KEYS[1], 'owner') ~= ARGV[1]
   or redis.call('HGET', KEYS[1], 'lease_token') ~= ARGV[2] then
    return 0
end
redis.call('HSET', KEYS[1],
    'status', 'completed',
    'result', ARGV[3],
    'completed_at', ARGV[4],
    'updated_at', ARGV[4],
    'completed_by', ARGV[1],
    'owner', '',
    'lease_token', '',
    'lease_until', '')
redis.call('ZREM', KEYS[2], ARGV[5])
redis.call('XADD', KEYS[3], 'MAXLEN', '~', ARGV[6], '*',
    'task_id', ARGV[5], 'status', 'completed', 'owner', ARGV[1], 'timestamp', ARGV[4])
redis.call('EXPIRE', KEYS[1], ARGV[7])
return 1
"""

    _FAIL_SCRIPT = """
if redis.call('HGET', KEYS[1], 'status') ~= 'running'
   or redis.call('HGET', KEYS[1], 'owner') ~= ARGV[1]
   or redis.call('HGET', KEYS[1], 'lease_token') ~= ARGV[2] then
    return 'stale'
end
local retries = tonumber(redis.call('HINCRBY', KEYS[1], 'retry_count', 1))
local max_retries = tonumber(redis.call('HGET', KEYS[1], 'max_retries'))
redis.call('ZREM', KEYS[2], ARGV[3])
redis.call('HSET', KEYS[1],
    'error', ARGV[4],
    'updated_at', ARGV[5],
    'owner', '',
    'lease_token', '',
    'lease_until', '')
if retries <= max_retries then
    redis.call('HSET', KEYS[1], 'status', 'pending')
    redis.call('ZADD', KEYS[3], ARGV[6], ARGV[3])
    return 'retry'
end
redis.call('HSET', KEYS[1], 'status', 'failed', 'completed_at', ARGV[5])
redis.call('XADD', KEYS[4], 'MAXLEN', '~', ARGV[7], '*',
    'task_id', ARGV[3], 'status', 'failed', 'owner', ARGV[1], 'timestamp', ARGV[5])
redis.call('EXPIRE', KEYS[1], ARGV[8])
return 'failed'
"""

    _CANCEL_SCRIPT = """
local status = redis.call('HGET', KEYS[1], 'status')
if not status
   or status == 'completed'
   or status == 'failed'
   or status == 'cancelled' then
    return 0
end
redis.call('ZREM', KEYS[2], ARGV[1])
redis.call('ZREM', KEYS[3], ARGV[1])
redis.call('HSET', KEYS[1],
    'status', 'cancelled',
    'completed_at', ARGV[2],
    'updated_at', ARGV[2],
    'owner', '',
    'lease_token', '',
    'lease_until', '')
redis.call('XADD', KEYS[4], 'MAXLEN', '~', ARGV[3], '*',
    'task_id', ARGV[1], 'status', 'cancelled', 'timestamp', ARGV[2])
redis.call('EXPIRE', KEYS[1], ARGV[4])
return 1
"""

    _RECLAIM_SCRIPT = """
local ids = redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', ARGV[1], 'LIMIT', 0, ARGV[2])
local reclaimed = 0
for _, id in ipairs(ids) do
    if redis.call('ZREM', KEYS[1], id) == 1 then
        local task_key = ARGV[3] .. id
        if redis.call('HGET', task_key, 'status') == 'running' then
            local retries = tonumber(redis.call('HINCRBY', task_key, 'retry_count', 1))
            local max_retries = tonumber(redis.call('HGET', task_key, 'max_retries'))
            if retries <= max_retries then
                local priority = tonumber(redis.call('HGET', task_key, 'priority'))
                redis.call('HSET', task_key,
                    'status', 'pending',
                    'error', 'worker lease expired',
                    'owner', '',
                    'lease_token', '',
                    'lease_until', '',
                    'updated_at', ARGV[4])
                redis.call('ZADD', KEYS[2], -priority, id)
                reclaimed = reclaimed + 1
            else
                redis.call('HSET', task_key,
                    'status', 'failed',
                    'error', 'worker lease expired',
                    'completed_at', ARGV[4],
                    'updated_at', ARGV[4],
                    'owner', '',
                    'lease_token', '',
                    'lease_until', '')
                redis.call('EXPIRE', task_key, ARGV[5])
            end
        end
    end
end
return reclaimed
"""

    def __init__(self) -> None:
        config = get_config()
        queue_config = config.task_queue
        prefix = queue_config.key_prefix.rstrip(":")
        self.prefix = f"{prefix}:"
        self.pending_key = f"{self.prefix}tasks:pending"
        self.processing_key = f"{self.prefix}tasks:processing"
        self.index_key = f"{self.prefix}tasks:index"
        self.history_key = f"{self.prefix}tasks:history"
        self.task_prefix = f"{self.prefix}task:"
        self.worker_key = f"{self.prefix}workers"
        self.config = queue_config
        self.terminal_ttl = max(
            self.config.metadata_ttl, self.config.result_ttl
        )
        self.redis = None
        self._last_reclaim_at = 0.0
        self._worker_heartbeats: Dict[str, float] = {}

    async def initialize(self) -> None:
        await redis_manager.initialize()
        self.redis = redis_manager.redis
        if self.redis is None:
            raise RuntimeError("Redis client is unavailable")

    def _task_key(self, task_id: str) -> str:
        return f"{self.task_prefix}{task_id}"

    def _events_key(self, task_id: str) -> str:
        return f"{self.task_prefix}{task_id}:events"

    @staticmethod
    def _encode_task(task: Dict[str, Any]) -> Dict[str, str]:
        json_fields = {"input_data", "result"}
        mapping: Dict[str, str] = {}
        for key, value in task.items():
            if key == "events":
                continue
            if key in json_fields:
                mapping[key] = "" if value is None else json.dumps(
                    value, ensure_ascii=False, default=str
                )
            elif value is None:
                mapping[key] = ""
            else:
                mapping[key] = str(value)
        return mapping

    @staticmethod
    def _decode_mapping(data: Dict[str, str]) -> Optional[Dict[str, Any]]:
        if not data:
            return None
        task: Dict[str, Any] = dict(data)
        for field in ("input_data", "result"):
            task[field] = json.loads(task[field]) if task.get(field) else None
        for field in ("priority", "retry_count", "max_retries"):
            task[field] = int(task.get(field) or 0)
        task["lease_until"] = (
            float(task["lease_until"]) if task.get("lease_until") else None
        )
        for field in (
            "owner", "completed_by", "lease_token", "error",
            "started_at", "completed_at",
        ):
            task[field] = task.get(field) or None
        return task

    async def _decode_task(
        self, task_id: str, include_events: bool = True
    ) -> Optional[Dict[str, Any]]:
        data = await self.redis.hgetall(self._task_key(task_id))
        task = self._decode_mapping(data)
        if not task:
            return None
        if include_events:
            values = await self.redis.lrange(self._events_key(task_id), 0, -1)
            task["events"] = [json.loads(value) for value in values]
        return task

    async def submit(self, task: Dict[str, Any]) -> None:
        task_id = task["id"]
        task_key = self._task_key(task_id)
        created = await self.redis.eval(
            self._SUBMIT_SCRIPT,
            3,
            task_key,
            self.pending_key,
            self.index_key,
            -int(task["priority"]),
            task_id,
            time.time(),
            json.dumps(self._encode_task(task), ensure_ascii=False),
        )
        if not created:
            raise ValueError(f"Task already exists: {task_id}")

    async def _reclaim_expired(self) -> None:
        now_monotonic = time.monotonic()
        reclaim_interval = max(1.0, min(self.config.lease_seconds / 2, 10.0))
        if now_monotonic - self._last_reclaim_at < reclaim_interval:
            return
        self._last_reclaim_at = now_monotonic
        reclaimed = await self.redis.eval(
            self._RECLAIM_SCRIPT,
            2,
            self.processing_key,
            self.pending_key,
            time.time(),
            100,
            self.task_prefix,
            utc_now(),
            self.terminal_ttl,
        )
        if reclaimed:
            logger.warning("重新入队 %s 个租约过期任务", reclaimed)

    async def claim_next(self, worker_id: str) -> Optional[Tuple[Dict[str, Any], str]]:
        await self._reclaim_expired()
        token = uuid4().hex
        now = utc_now()
        lease_until = time.time() + self.config.lease_seconds
        task_id = await self.redis.eval(
            self._CLAIM_NEXT_SCRIPT,
            2,
            self.pending_key,
            self.processing_key,
            self.task_prefix,
            worker_id,
            token,
            lease_until,
            now,
        )
        if not task_id:
            return None
        task = await self._decode_task(task_id)
        return (task, token) if task else None

    async def claim_task(
        self, task_id: str, worker_id: str
    ) -> Optional[Tuple[Dict[str, Any], str]]:
        token = uuid4().hex
        now = utc_now()
        lease_until = time.time() + self.config.lease_seconds
        claimed = await self.redis.eval(
            self._CLAIM_TASK_SCRIPT,
            2,
            self.pending_key,
            self.processing_key,
            task_id,
            self.task_prefix,
            worker_id,
            token,
            lease_until,
            now,
        )
        if not claimed:
            return None
        task = await self._decode_task(task_id)
        return (task, token) if task else None

    async def heartbeat(self, task_id: str, worker_id: str, lease_token: str) -> bool:
        lease_until = time.time() + self.config.lease_seconds
        result = await self.redis.eval(
            self._HEARTBEAT_SCRIPT,
            2,
            self._task_key(task_id),
            self.processing_key,
            worker_id,
            lease_token,
            lease_until,
            utc_now(),
            task_id,
        )
        return bool(result)

    async def complete(
        self, task_id: str, worker_id: str, lease_token: str, result: Dict[str, Any]
    ) -> bool:
        completed_at = utc_now()
        updated = await self.redis.eval(
            self._COMPLETE_SCRIPT,
            3,
            self._task_key(task_id),
            self.processing_key,
            self.history_key,
            worker_id,
            lease_token,
            json.dumps(result, ensure_ascii=False, default=str),
            completed_at,
            task_id,
            self.config.history_maxlen,
            self.terminal_ttl,
        )
        return bool(updated)

    async def fail(
        self, task_id: str, worker_id: str, lease_token: str, error: str
    ) -> str:
        task = await self._decode_task(task_id, include_events=False)
        if not task:
            return "failed"
        outcome = await self.redis.eval(
            self._FAIL_SCRIPT,
            4,
            self._task_key(task_id),
            self.processing_key,
            self.pending_key,
            self.history_key,
            worker_id,
            lease_token,
            task_id,
            error,
            utc_now(),
            -int(task["priority"]),
            self.config.history_maxlen,
            self.terminal_ttl,
        )
        return outcome or "failed"

    async def cancel(self, task_id: str) -> bool:
        result = await self.redis.eval(
            self._CANCEL_SCRIPT,
            4,
            self._task_key(task_id),
            self.pending_key,
            self.processing_key,
            self.history_key,
            task_id,
            utc_now(),
            self.config.history_maxlen,
            self.terminal_ttl,
        )
        return bool(result)

    async def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        return await self._decode_task(task_id)

    async def list(self, limit: int = 50) -> List[Dict[str, Any]]:
        task_ids = await self.redis.zrevrange(self.index_key, 0, max(limit * 2, 20))
        if not task_ids:
            return []
        async with self.redis.pipeline(transaction=False) as pipe:
            for task_id in task_ids:
                pipe.hgetall(self._task_key(task_id))
            task_rows = await pipe.execute()
        tasks = []
        stale_ids = []
        for task_id, task_data in zip(task_ids, task_rows):
            task = self._decode_mapping(task_data)
            if task:
                tasks.append(task)
                if len(tasks) >= limit:
                    break
            else:
                stale_ids.append(task_id)
        if stale_ids:
            await self.redis.zrem(self.index_key, *stale_ids)
        return tasks

    async def append_event(self, task_id: str, event: Dict[str, Any]) -> None:
        key = self._events_key(task_id)
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.rpush(key, json.dumps(event, ensure_ascii=False, default=str))
            pipe.ltrim(key, -self.config.history_maxlen, -1)
            pipe.expire(key, self.terminal_ttl)
            pipe.hset(self._task_key(task_id), "updated_at", utc_now())
            await pipe.execute()

    async def queue_size(self) -> int:
        return await self.redis.zcard(self.pending_key)

    async def heartbeat_worker(self, worker_id: str) -> None:
        now_monotonic = time.monotonic()
        last_heartbeat = self._worker_heartbeats.get(worker_id, 0.0)
        if now_monotonic - last_heartbeat < self.config.heartbeat_seconds:
            return
        await self.redis.set(
            f"{self.worker_key}:{worker_id}",
            time.time(),
            ex=max(self.config.lease_seconds * 2, 60),
        )
        self._worker_heartbeats[worker_id] = now_monotonic

    async def close(self) -> None:
        await redis_manager.close()


def create_task_backend() -> TaskQueueBackend:
    backend = get_config().task_queue.backend
    if backend == "local":
        return MemoryTaskQueueBackend()
    if backend == "redis":
        return RedisTaskQueueBackend()
    raise ValueError(f"Unsupported TASK_QUEUE_BACKEND={backend!r}; use local or redis")
