import asyncio
import unittest

from core.concurrency import ConcurrencyLimiter


class LocalConcurrencyLimiterTest(unittest.IsolatedAsyncioTestCase):
    async def test_scope_never_exceeds_limit(self):
        limiter = ConcurrencyLimiter()
        active = 0
        maximum = 0
        lock = asyncio.Lock()

        async def work():
            nonlocal active, maximum
            async with limiter.slot("user:tenant:user", 2):
                async with lock:
                    active += 1
                    maximum = max(maximum, active)
                await asyncio.sleep(0.01)
                async with lock:
                    active -= 1

        await asyncio.gather(*(work() for _ in range(8)))
        self.assertEqual(maximum, 2)


if __name__ == "__main__":
    unittest.main()
