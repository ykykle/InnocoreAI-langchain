import asyncio
import os
import unittest
from uuid import uuid4

from core.config import get_config
from core.task_queue import RedisTaskQueueBackend
from tests.test_task_queue import make_task


@unittest.skipUnless(
    os.getenv("RUN_REDIS_INTEGRATION") == "1",
    "set RUN_REDIS_INTEGRATION=1 to run against a disposable Redis database",
)
class RedisTaskQueueIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        config = get_config()
        self.original_prefix = config.task_queue.key_prefix
        self.original_lease = config.task_queue.lease_seconds
        self.original_port = config.redis.port
        config.task_queue.key_prefix = f"innocore:test:{uuid4().hex}"
        config.task_queue.lease_seconds = 1
        config.redis.port = int(os.getenv("REDIS_TEST_PORT", str(config.redis.port)))
        self.backend = RedisTaskQueueBackend()
        await self.backend.initialize()

    async def asyncTearDown(self):
        keys = [
            key
            async for key in self.backend.redis.scan_iter(
                match=f"{self.backend.prefix}*"
            )
        ]
        if keys:
            await self.backend.redis.delete(*keys)
        config = get_config()
        config.task_queue.key_prefix = self.original_prefix
        config.task_queue.lease_seconds = self.original_lease
        config.redis.port = self.original_port
        await self.backend.close()

    async def test_concurrent_workers_only_claim_once(self):
        await self.backend.submit(make_task("redis-once"))

        first, second = await asyncio.gather(
            self.backend.claim_task("redis-once", "worker-a"),
            self.backend.claim_task("redis-once", "worker-b"),
        )

        self.assertEqual(sum(value is not None for value in (first, second)), 1)

    async def test_expired_lease_is_reclaimed(self):
        await self.backend.submit(make_task("redis-reclaim", max_retries=2))
        claimed = await self.backend.claim_task("redis-reclaim", "worker-a")
        self.assertIsNotNone(claimed)

        await asyncio.sleep(1.1)
        reclaimed = await self.backend.claim_next("worker-b")

        self.assertIsNotNone(reclaimed)
        task, _ = reclaimed
        self.assertEqual(task["id"], "redis-reclaim")
        self.assertEqual(task["retry_count"], 1)
        self.assertEqual(task["owner"], "worker-b")

    async def test_priority_completion_and_fencing(self):
        await self.backend.submit(make_task("redis-low", priority=1))
        await self.backend.submit(make_task("redis-high", priority=9))

        task, token = await self.backend.claim_next("worker-a")
        self.assertEqual(task["id"], "redis-high")
        self.assertFalse(
            await self.backend.complete(
                task["id"], "worker-a", "stale-token", {"ok": False}
            )
        )
        self.assertTrue(
            await self.backend.complete(
                task["id"], "worker-a", token, {"ok": True}
            )
        )
        stored = await self.backend.get(task["id"])
        self.assertEqual(stored["status"], "completed")
        self.assertEqual(stored["result"], {"ok": True})

    async def test_cancel_removes_pending_task(self):
        await self.backend.submit(make_task("redis-cancel"))

        self.assertTrue(await self.backend.cancel("redis-cancel"))
        self.assertIsNone(
            await self.backend.claim_task("redis-cancel", "worker-a")
        )
        self.assertEqual(
            (await self.backend.get("redis-cancel"))["status"], "cancelled"
        )

    async def test_failure_retries_then_becomes_terminal(self):
        await self.backend.submit(make_task("redis-retry", max_retries=1))
        task, token = await self.backend.claim_task("redis-retry", "worker-a")

        self.assertEqual(
            await self.backend.fail(
                task["id"], "worker-a", token, "temporary failure"
            ),
            "retry",
        )
        retried, retry_token = await self.backend.claim_next("worker-b")
        self.assertEqual(retried["retry_count"], 1)
        self.assertEqual(
            await self.backend.fail(
                retried["id"], "worker-b", retry_token, "permanent failure"
            ),
            "failed",
        )
        stored = await self.backend.get("redis-retry")
        self.assertEqual(stored["status"], "failed")
        self.assertEqual(stored["retry_count"], 2)


if __name__ == "__main__":
    unittest.main()
