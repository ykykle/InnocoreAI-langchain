import asyncio
import os
import unittest
from unittest.mock import patch

from core.config import InnoCoreConfig, get_config
from core.task_queue import MemoryTaskQueueBackend, utc_now


def make_task(task_id: str, priority: int = 0, max_retries: int = 1):
    return {
        "id": task_id,
        "type": "paper_analysis",
        "input_data": {"paper_id": task_id},
        "status": "pending",
        "priority": priority,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "started_at": None,
        "completed_at": None,
        "result": None,
        "error": None,
        "owner": None,
        "lease_token": None,
        "lease_until": None,
        "retry_count": 0,
        "max_retries": max_retries,
        "events": [],
    }


class MemoryTaskQueueBackendTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.backend = MemoryTaskQueueBackend()
        await self.backend.initialize()

    async def test_higher_priority_is_claimed_first(self):
        await self.backend.submit(make_task("low", priority=1))
        await self.backend.submit(make_task("high", priority=10))

        task, _ = await self.backend.claim_next("worker-a")

        self.assertEqual(task["id"], "high")

    async def test_task_can_only_be_claimed_once(self):
        await self.backend.submit(make_task("once"))

        first, second = await asyncio.gather(
            self.backend.claim_task("once", "worker-a"),
            self.backend.claim_task("once", "worker-b"),
        )

        self.assertEqual(sum(value is not None for value in (first, second)), 1)

    async def test_fencing_token_rejects_stale_completion(self):
        await self.backend.submit(make_task("fenced"))
        task, token = await self.backend.claim_task("fenced", "worker-a")

        stale = await self.backend.complete(
            task["id"], "worker-a", "stale-token", {"ok": False}
        )
        current = await self.backend.get(task["id"])
        committed = await self.backend.complete(
            task["id"], "worker-a", token, {"ok": True}
        )

        self.assertFalse(stale)
        self.assertEqual(current["status"], "running")
        self.assertTrue(committed)
        self.assertEqual((await self.backend.get(task["id"]))["result"], {"ok": True})

    async def test_failure_retries_then_becomes_terminal(self):
        await self.backend.submit(make_task("retry", max_retries=1))
        task, token = await self.backend.claim_task("retry", "worker-a")

        first = await self.backend.fail(task["id"], "worker-a", token, "temporary")
        retried, second_token = await self.backend.claim_next("worker-b")
        second = await self.backend.fail(
            retried["id"], "worker-b", second_token, "permanent"
        )

        self.assertEqual(first, "retry")
        self.assertEqual(second, "failed")
        self.assertEqual((await self.backend.get("retry"))["status"], "failed")

    async def test_cancelled_task_cannot_be_claimed(self):
        await self.backend.submit(make_task("cancel"))

        self.assertTrue(await self.backend.cancel("cancel"))
        self.assertIsNone(await self.backend.claim_task("cancel", "worker-a"))
        self.assertEqual((await self.backend.get("cancel"))["status"], "cancelled")

    async def test_event_and_result_are_available_to_other_readers(self):
        await self.backend.submit(make_task("events"))
        await self.backend.append_event(
            "events", {"time": utc_now(), "type": "progress", "message": "half"}
        )
        task, token = await self.backend.claim_task("events", "worker-a")
        await self.backend.complete(task["id"], "worker-a", token, {"answer": "done"})

        stored = await self.backend.get("events")
        self.assertEqual(stored["events"][0]["message"], "half")
        self.assertEqual(stored["result"]["answer"], "done")

    async def test_default_mode_remains_local(self):
        self.assertEqual(get_config().task_queue.backend, "local")

    async def test_environment_selects_distributed_mode_and_redis_endpoint(self):
        environment = {
            "TASK_QUEUE_BACKEND": "redis",
            "TASK_WORKER_ENABLED": "false",
            "TASK_QUEUE_KEY_PREFIX": "innocore:test-env",
            "REDIS_HOST": "redis.internal",
            "REDIS_PORT": "6380",
            "REDIS_DB": "4",
        }
        with patch.dict(os.environ, environment, clear=False):
            config = InnoCoreConfig()

        self.assertEqual(config.task_queue.backend, "redis")
        self.assertFalse(config.task_queue.worker_enabled)
        self.assertEqual(config.task_queue.key_prefix, "innocore:test-env")
        self.assertEqual(config.redis.host, "redis.internal")
        self.assertEqual(config.redis.port, 6380)
        self.assertEqual(config.redis.db, 4)


if __name__ == "__main__":
    unittest.main()
