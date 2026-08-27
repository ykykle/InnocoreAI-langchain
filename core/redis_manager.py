"""
Redis 管理器 - 任务队列 + 缓存 + 会话状态 + Agent Checkpoint
"""

import json
import logging
from typing import Any, Dict, List, Optional

try:
    import redis.asyncio as aioredis
    HAS_REDIS = True
except ImportError:
    aioredis = None
    HAS_REDIS = False

from .config import get_config

logger = logging.getLogger(__name__)


class RedisManager:
    """Redis 连接管理器，提供任务队列、缓存、会话状态功能"""

    def __init__(self):
        self.config = get_config().redis
        self.redis: Optional[Any] = None
        self._initialize_lock = None

    async def initialize(self):
        if not HAS_REDIS:
            raise RuntimeError("redis 包未安装，Redis 功能不可用")

        # asyncio primitives should be created in the active event loop.
        if self._initialize_lock is None:
            import asyncio
            self._initialize_lock = asyncio.Lock()

        async with self._initialize_lock:
            if self.redis is not None:
                await self.redis.ping()
                return

            common_options = {
                "max_connections": self.config.max_connections,
                "decode_responses": True,
                "socket_connect_timeout": self.config.socket_connect_timeout,
                "socket_timeout": self.config.socket_timeout,
                "health_check_interval": self.config.health_check_interval,
            }
            if self.config.url:
                client = aioredis.from_url(self.config.url, **common_options)
            else:
                client = aioredis.Redis(
                    host=self.config.host,
                    port=self.config.port,
                    db=self.config.db,
                    password=self.config.password,
                    **common_options,
                )
            try:
                await client.ping()
            except Exception:
                await client.aclose()
                raise
            self.redis = client
            logger.info(
                "Redis 初始化完成: %s:%s/%s",
                self.config.host,
                self.config.port,
                self.config.db,
            )

    def _ensure_redis(self):
        """确保 Redis 可用"""
        if not self.redis:
            raise RuntimeError("Redis 未初始化或不可用")

    # ---- 旧版 Sorted Set 辅助方法（新任务调度使用 Redis Stream） ----
    async def push_task(self, queue: str, task_id: str, priority: int = 0) -> None:
        self._ensure_redis()
        await self.redis.zadd(queue, {task_id: -priority})

    async def pop_task(self, queue: str) -> Optional[str]:
        self._ensure_redis()
        result = await self.redis.zpopmin(queue)
        return result[0][0] if result else None

    async def task_queue_size(self, queue: str) -> int:
        self._ensure_redis()
        return await self.redis.zcard(queue)

    # ---- 活跃任务 (Hash) ----
    async def set_active_task(self, task_id: str, task_data: Dict) -> None:
        if not self.redis:
            return
        await self.redis.hset("active_tasks", task_id, json.dumps(task_data, ensure_ascii=False))

    async def get_active_task(self, task_id: str) -> Optional[Dict]:
        if not self.redis:
            return None
        val = await self.redis.hget("active_tasks", task_id)
        return json.loads(val) if val else None

    async def remove_active_task(self, task_id: str) -> None:
        if not self.redis:
            return
        await self.redis.hdel("active_tasks", task_id)

    # ---- 任务历史 (List，保留最近 1000 条) ----
    async def push_task_history(self, task_data: Dict) -> None:
        if not self.redis:
            return
        await self.redis.lpush("task_history", json.dumps(task_data, ensure_ascii=False))
        await self.redis.ltrim("task_history", 0, 999)

    async def get_task_history(self, limit: int = 50) -> List[Dict]:
        if not self.redis:
            return []
        items = await self.redis.lrange("task_history", 0, limit - 1)
        return [json.loads(item) for item in items]

    # ---- 通用缓存 (String + TTL) ----
    async def cache_set(self, key: str, value: Any, ttl: int = 3600) -> None:
        if not self.redis:
            return
        await self.redis.set(key, json.dumps(value, ensure_ascii=False), ex=ttl)

    async def cache_get(self, key: str) -> Optional[Any]:
        if not self.redis:
            return None
        val = await self.redis.get(key)
        return json.loads(val) if val else None

    async def cache_delete(self, key: str) -> None:
        if not self.redis:
            return
        await self.redis.delete(key)

    # ---- Agent 会话状态 (Hash + TTL) ----
    async def set_agent_state(self, agent_name: str, state: Dict, ttl: int = 86400) -> None:
        if not self.redis:
            return
        key = f"agent_state:{agent_name}"
        mapping = {k: json.dumps(v, ensure_ascii=False) for k, v in state.items()}
        await self.redis.hset(key, mapping=mapping)
        await self.redis.expire(key, ttl)

    async def get_agent_state(self, agent_name: str) -> Optional[Dict]:
        if not self.redis:
            return None
        data = await self.redis.hgetall(f"agent_state:{agent_name}")
        return {k: json.loads(v) for k, v in data.items()} if data else None

    # ---- Pub/Sub Agent 间事件 ----
    async def publish(self, channel: str, message: Dict) -> None:
        if not self.redis:
            return
        await self.redis.publish(channel, json.dumps(message, ensure_ascii=False))

    async def close(self):
        if self.redis:
            await self.redis.aclose()
            self.redis = None
            logger.info("Redis 连接已关闭")


redis_manager = RedisManager()
