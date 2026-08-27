"""Lease-based concurrency limits shared by all application instances."""

import asyncio
import hashlib
import time
from contextlib import asynccontextmanager, suppress
from typing import AsyncIterator, Dict
from uuid import uuid4

from .config import get_config
from .redis_manager import redis_manager


_ACQUIRE = """
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', ARGV[1])
if redis.call('ZCARD', KEYS[1]) >= tonumber(ARGV[2]) then return 0 end
redis.call('ZADD', KEYS[1], ARGV[3], ARGV[4])
redis.call('EXPIRE', KEYS[1], ARGV[5])
return 1
"""


class ConcurrencyLimiter:
    """Uses local semaphores in development and Redis leases in production."""

    def __init__(self) -> None:
        self.config = get_config().task_queue
        self._local: Dict[str, asyncio.Semaphore] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _safe_scope(scope: str) -> str:
        return hashlib.sha256(scope.encode("utf-8")).hexdigest()[:24]

    async def _local_semaphore(self, scope: str, limit: int) -> asyncio.Semaphore:
        async with self._lock:
            return self._local.setdefault(scope, asyncio.Semaphore(limit))

    async def _renew(self, key: str, token: str, lease_seconds: int) -> None:
        while True:
            await asyncio.sleep(max(1, lease_seconds // 3))
            redis = redis_manager.redis
            if redis is None:
                return
            score = time.time() + lease_seconds
            # XX prevents a released/expired token from being recreated.
            await redis.zadd(key, {token: score}, xx=True)

    @asynccontextmanager
    async def slot(self, scope: str, limit: int) -> AsyncIterator[None]:
        if limit < 1:
            raise ValueError("concurrency limit must be positive")
        if self.config.backend != "redis_stream":
            semaphore = await self._local_semaphore(scope, limit)
            async with semaphore:
                yield
            return

        await redis_manager.initialize()
        redis = redis_manager.redis
        key = (
            f"{self.config.key_prefix.rstrip(':')}:concurrency:"
            f"{self._safe_scope(scope)}"
        )
        token = uuid4().hex
        lease_seconds = max(self.config.lease_seconds, 30)
        while True:
            now = time.time()
            acquired = await redis.eval(
                _ACQUIRE, 1, key, now, limit, now + lease_seconds,
                token, lease_seconds * 2,
            )
            if acquired:
                break
            await asyncio.sleep(max(0.05, self.config.poll_interval_ms / 1000))

        renewer = asyncio.create_task(self._renew(key, token, lease_seconds))
        try:
            yield
        finally:
            renewer.cancel()
            with suppress(asyncio.CancelledError):
                await renewer
            await redis.zrem(key, token)


concurrency_limiter = ConcurrencyLimiter()
