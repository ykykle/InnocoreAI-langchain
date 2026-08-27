"""Opt-in integration tests requiring disposable PostgreSQL and Redis."""

import asyncio
import os
import unittest
from uuid import uuid4

from core.config import get_config
from core.database import db_manager
from core.task_queue import RedisTaskQueueBackend
from tests.test_task_queue import make_task


@unittest.skipUnless(
    os.getenv("RUN_DISTRIBUTED_INTEGRATION") == "1",
    "set RUN_DISTRIBUTED_INTEGRATION=1 with disposable PostgreSQL and Redis",
)
class RedisStreamTaskQueueIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        config = get_config()
        self.original = (
            config.task_queue.key_prefix,
            config.task_queue.lease_seconds,
            config.task_queue.stream_claim_idle_ms,
            config.redis.port,
        )
        self.test_tenant = f"integration-{uuid4().hex}"
        config.task_queue.key_prefix = f"innocore:test:{uuid4().hex}"
        config.task_queue.lease_seconds = 1
        config.task_queue.stream_claim_idle_ms = 1000
        config.redis.port = int(os.getenv("REDIS_TEST_PORT", str(config.redis.port)))
        self.backend = RedisTaskQueueBackend()
        await self.backend.initialize()

    def task(self, task_id: str, max_retries: int = 1):
        task = make_task(task_id, max_retries=max_retries)
        task["tenant_id"] = self.test_tenant
        return task

    async def asyncTearDown(self):
        await self.backend.redis.delete(self.backend.stream_key)
        async with db_manager.get_connection() as conn:
            await conn.execute(
                "DELETE FROM agent_tasks WHERE tenant_id=$1", self.test_tenant
            )
        config = get_config()
        (
            config.task_queue.key_prefix,
            config.task_queue.lease_seconds,
            config.task_queue.stream_claim_idle_ms,
            config.redis.port,
        ) = self.original
        await self.backend.close()

    async def test_consumer_group_and_fencing(self):
        await self.backend.submit(self.task("stream-once"))
        claimed = await self.backend.claim_next("worker-a")
        self.assertIsNotNone(claimed)
        task, token = claimed
        self.assertFalse(
            await self.backend.complete(task["id"], "worker-a", "stale", {})
        )
        self.assertTrue(
            await self.backend.complete(task["id"], "worker-a", token, {"ok": True})
        )
        self.assertEqual((await self.backend.get(task["id"]))["result"], {"ok": True})

    async def test_expired_delivery_is_reclaimed(self):
        await self.backend.submit(self.task("stream-reclaim", max_retries=2))
        self.assertIsNotNone(await self.backend.claim_next("worker-a"))
        await asyncio.sleep(1.1)
        task, _ = await self.backend.claim_next("worker-b")
        self.assertEqual(task["id"], "stream-reclaim")
        self.assertEqual(task["retry_count"], 1)

    async def test_cancel_pending_task(self):
        await self.backend.submit(self.task("stream-cancel"))
        self.assertTrue(await self.backend.cancel("stream-cancel"))
        self.assertIsNone(await self.backend.claim_next("worker-a"))
        self.assertEqual(
            (await self.backend.get("stream-cancel"))["status"], "cancelled"
        )


if __name__ == "__main__":
    unittest.main()
