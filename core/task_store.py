"""PostgreSQL-backed task state machine used by the distributed queue.

Redis only transports task identifiers.  Every authoritative transition is a
compare-and-set operation in PostgreSQL and is accompanied by an audit event.
"""

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .database import db_manager


TERMINAL_STATUSES = {"completed", "failed", "cancelled", "timed_out"}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _iso(value: Any) -> Optional[str]:
    return value.isoformat() if value else None


def _decode_json(value: Any, default: Any) -> Any:
    if value is None:
        return default
    return json.loads(value) if isinstance(value, str) else value


class PostgresTaskStore:
    """Durable task state, leases, audit events and transactional outbox."""

    async def initialize(self) -> None:
        async with db_manager.get_connection() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_tasks (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    task_type VARCHAR(50) NOT NULL,
                    agent_type VARCHAR(30) NOT NULL,
                    workflow_id TEXT,
                    parent_task_id TEXT,
                    status VARCHAR(30) NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 0,
                    input_data JSONB NOT NULL DEFAULT '{}',
                    result JSONB,
                    error TEXT,
                    owner TEXT,
                    execution_token TEXT,
                    stream_message_id TEXT,
                    lease_until TIMESTAMPTZ,
                    heartbeat_at TIMESTAMPTZ,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    max_retries INTEGER NOT NULL DEFAULT 3,
                    next_run_at TIMESTAMPTZ,
                    cancel_requested_at TIMESTAMPTZ,
                    version INTEGER NOT NULL DEFAULT 0,
                    idempotency_key TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    started_at TIMESTAMPTZ,
                    completed_at TIMESTAMPTZ,
                    completed_by TEXT,
                    UNIQUE (tenant_id, user_id, idempotency_key)
                );

                CREATE TABLE IF NOT EXISTS agent_task_events (
                    id BIGSERIAL PRIMARY KEY,
                    task_id TEXT NOT NULL REFERENCES agent_tasks(id) ON DELETE CASCADE,
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    event_type VARCHAR(50) NOT NULL,
                    from_status VARCHAR(30),
                    to_status VARCHAR(30),
                    payload JSONB NOT NULL DEFAULT '{}',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE(task_id, version)
                );

                CREATE TABLE IF NOT EXISTS agent_task_outbox (
                    id BIGSERIAL PRIMARY KEY,
                    task_id TEXT NOT NULL REFERENCES agent_tasks(id) ON DELETE CASCADE,
                    event_key TEXT NOT NULL UNIQUE,
                    topic VARCHAR(100) NOT NULL,
                    payload JSONB NOT NULL,
                    available_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    locked_by TEXT,
                    locked_until TIMESTAMPTZ,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    published_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );

                CREATE INDEX IF NOT EXISTS idx_agent_tasks_owner_status
                    ON agent_tasks(tenant_id, user_id, status);
                CREATE INDEX IF NOT EXISTS idx_agent_tasks_agent_status
                    ON agent_tasks(agent_type, status);
                CREATE INDEX IF NOT EXISTS idx_agent_tasks_lease
                    ON agent_tasks(status, lease_until);
                CREATE INDEX IF NOT EXISTS idx_agent_task_events_task
                    ON agent_task_events(task_id, version);
                CREATE INDEX IF NOT EXISTS idx_agent_task_outbox_ready
                    ON agent_task_outbox(published_at, available_at, locked_until);
                """
            )

    @staticmethod
    def _row(row: Any) -> Optional[Dict[str, Any]]:
        if not row:
            return None
        item = dict(row)
        item["type"] = item.pop("task_type")
        item["input_data"] = _decode_json(item.get("input_data"), {})
        item["result"] = _decode_json(item.get("result"), None)
        item["lease_token"] = item.pop("execution_token", None)
        for key in (
            "created_at", "updated_at", "started_at", "completed_at",
            "lease_until", "heartbeat_at", "next_run_at", "cancel_requested_at",
        ):
            item[key] = _iso(item.get(key))
        return item

    async def create(self, task: Dict[str, Any], topic: str) -> None:
        async with db_manager.get_connection() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO agent_tasks (
                        id, tenant_id, user_id, task_type, agent_type,
                        workflow_id, parent_task_id, status, priority,
                        input_data, max_retries, idempotency_key
                    ) VALUES ($1,$2,$3,$4,$5,$6,$7,'pending',$8,$9::jsonb,$10,$11)
                    """,
                    task["id"], task["tenant_id"], task["user_id"], task["type"],
                    task["agent_type"], task.get("workflow_id"),
                    task.get("parent_task_id"), int(task.get("priority", 0)),
                    _json(task.get("input_data") or {}), int(task.get("max_retries", 3)),
                    task.get("idempotency_key"),
                )
                await self._event(
                    conn, task["id"], task["tenant_id"], task["user_id"], 0,
                    "submitted", None, "pending", {},
                )
                await self._outbox(
                    conn, task["id"], f"{task['id']}:dispatch:0", topic,
                    {"task_id": task["id"], "version": 0}, datetime.now(timezone.utc),
                )

    async def claim_outbox(self, owner: str, limit: int) -> List[Dict[str, Any]]:
        async with db_manager.get_connection() as conn:
            rows = await conn.fetch(
                """
                WITH selected AS (
                    SELECT id FROM agent_task_outbox
                    WHERE published_at IS NULL
                      AND available_at <= NOW()
                      AND (locked_until IS NULL OR locked_until < NOW())
                    ORDER BY id
                    FOR UPDATE SKIP LOCKED
                    LIMIT $2
                )
                UPDATE agent_task_outbox o
                SET locked_by=$1, locked_until=NOW()+INTERVAL '30 seconds',
                    attempts=attempts+1
                FROM selected s WHERE o.id=s.id
                RETURNING o.*
                """,
                owner, limit,
            )
            return [dict(row) for row in rows]

    async def mark_outbox_published(self, outbox_id: int) -> None:
        async with db_manager.get_connection() as conn:
            await conn.execute(
                """UPDATE agent_task_outbox
                   SET published_at=NOW(), locked_by=NULL, locked_until=NULL
                   WHERE id=$1""",
                outbox_id,
            )

    async def release_outbox(self, outbox_id: int) -> None:
        async with db_manager.get_connection() as conn:
            await conn.execute(
                """UPDATE agent_task_outbox
                   SET locked_by=NULL, locked_until=NULL WHERE id=$1""",
                outbox_id,
            )

    async def defer_dispatch(
        self, task_id: str, topic: str, delay_seconds: float,
    ) -> bool:
        """Durably re-dispatch a quota-blocked task after a short delay."""
        async with db_manager.get_connection() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    "SELECT * FROM agent_tasks WHERE id=$1 FOR UPDATE", task_id
                )
                if not row or row["status"] not in {"pending", "retry_wait"}:
                    return False
                next_version = int(row["version"]) + 1
                available_at = datetime.now(timezone.utc) + timedelta(
                    seconds=max(0.1, delay_seconds)
                )
                await conn.execute(
                    """UPDATE agent_tasks SET version=$2, next_run_at=$3,
                       updated_at=NOW() WHERE id=$1""",
                    task_id, next_version, available_at,
                )
                await self._event(
                    conn, task_id, row["tenant_id"], row["user_id"], next_version,
                    "dispatch_deferred", row["status"], row["status"],
                    {"reason": "concurrency_limit"},
                )
                await self._outbox(
                    conn, task_id, f"{task_id}:dispatch:{next_version}", topic,
                    {"task_id": task_id, "version": next_version}, available_at,
                )
                return True

    async def claim(
        self, task_id: str, worker_id: str, token: str, message_id: str,
        lease_seconds: int, user_limit: int, miner_limit: int,
    ) -> Optional[Dict[str, Any]]:
        async with db_manager.get_connection() as conn:
            async with conn.transaction():
                # Serialize admission decisions across scheduler/worker instances.
                await conn.execute("SELECT pg_advisory_xact_lock(hashtext('innocore-task-admission'))")
                row = await conn.fetchrow(
                    "SELECT * FROM agent_tasks WHERE id=$1 FOR UPDATE", task_id
                )
                if not row:
                    return None
                current = dict(row)
                expired = (
                    current["status"] in {"running", "cancelling"}
                    and current.get("lease_until")
                    and current["lease_until"] < datetime.now(timezone.utc)
                )
                if current["status"] not in {"pending", "retry_wait"} and not expired:
                    return None
                if (
                    not expired
                    and current.get("next_run_at")
                    and current["next_run_at"] > datetime.now(timezone.utc)
                ):
                    return None
                if current.get("cancel_requested_at"):
                    next_version = int(current["version"]) + 1
                    await conn.execute(
                        """UPDATE agent_tasks SET status='cancelled', completed_at=NOW(),
                           updated_at=NOW(), version=$2 WHERE id=$1""",
                        task_id, next_version,
                    )
                    await self._event(
                        conn, task_id, current["tenant_id"], current["user_id"],
                        next_version, "cancelled", current["status"], "cancelled", {},
                    )
                    return None

                if expired and int(current["retry_count"]) + 1 > int(current["max_retries"]):
                    next_version = int(current["version"]) + 1
                    await conn.execute(
                        """UPDATE agent_tasks SET status='failed',
                           error='worker lease expired', completed_at=NOW(),
                           owner=NULL, execution_token=NULL, lease_until=NULL,
                           retry_count=retry_count+1, version=$2, updated_at=NOW()
                           WHERE id=$1""",
                        task_id, next_version,
                    )
                    await self._event(
                        conn, task_id, current["tenant_id"], current["user_id"],
                        next_version, "failed", "running", "failed",
                        {"error": "worker lease expired"},
                    )
                    return None

                active_user = await conn.fetchval(
                    """SELECT COUNT(*) FROM agent_tasks
                       WHERE tenant_id=$1 AND user_id=$2
                         AND status IN ('running','cancelling')
                         AND (lease_until IS NULL OR lease_until > NOW())
                         AND id<>$3""",
                    current["tenant_id"], current["user_id"], task_id,
                )
                if active_user >= user_limit:
                    return None
                if current["agent_type"] == "miner":
                    active_miner = await conn.fetchval(
                        """SELECT COUNT(*) FROM agent_tasks
                           WHERE agent_type='miner' AND status IN ('running','cancelling')
                             AND (lease_until IS NULL OR lease_until > NOW())
                             AND id<>$1""",
                        task_id,
                    )
                    if active_miner >= miner_limit:
                        return None

                old_status = current["status"]
                next_version = int(current["version"]) + 1
                claimed = await conn.fetchrow(
                    """
                    UPDATE agent_tasks SET
                        status='running', owner=$2, execution_token=$3,
                        stream_message_id=$4,
                        lease_until=NOW()+($5 * INTERVAL '1 second'),
                        heartbeat_at=NOW(), started_at=COALESCE(started_at,NOW()),
                        retry_count=CASE WHEN $6 THEN retry_count+1 ELSE retry_count END,
                        version=$7, updated_at=NOW()
                    WHERE id=$1 RETURNING *
                    """,
                    task_id, worker_id, token, message_id, lease_seconds,
                    bool(expired), next_version,
                )
                await self._event(
                    conn, task_id, current["tenant_id"], current["user_id"],
                    next_version, "claimed", old_status, "running",
                    {"worker_id": worker_id},
                )
                return self._row(claimed)

    async def heartbeat(
        self, task_id: str, worker_id: str, token: str, lease_seconds: int
    ) -> bool:
        async with db_manager.get_connection() as conn:
            result = await conn.execute(
                """UPDATE agent_tasks
                   SET lease_until=NOW()+($4 * INTERVAL '1 second'),
                       heartbeat_at=NOW(), updated_at=NOW()
                   WHERE id=$1 AND status='running' AND owner=$2
                     AND execution_token=$3""",
                task_id, worker_id, token, lease_seconds,
            )
            return result == "UPDATE 1"

    async def complete(
        self, task_id: str, worker_id: str, token: str, result: Dict[str, Any]
    ) -> bool:
        async with db_manager.get_connection() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    "SELECT * FROM agent_tasks WHERE id=$1 FOR UPDATE", task_id
                )
                if not row:
                    return False
                current = dict(row)
                if (
                    current["status"] == "cancelling"
                    and current.get("owner") == worker_id
                    and current.get("execution_token") == token
                ):
                    next_version = int(current["version"]) + 1
                    await conn.execute(
                        """UPDATE agent_tasks SET status='cancelled', completed_at=NOW(),
                           owner=NULL, execution_token=NULL, lease_until=NULL,
                           version=$2, updated_at=NOW() WHERE id=$1""",
                        task_id, next_version,
                    )
                    await self._event(
                        conn, task_id, current["tenant_id"], current["user_id"],
                        next_version, "cancelled", "cancelling", "cancelled", {},
                    )
                    return False
                if (
                    current["status"] != "running"
                    or current.get("owner") != worker_id
                    or current.get("execution_token") != token
                    or (current.get("lease_until") and current["lease_until"] <= datetime.now(timezone.utc))
                ):
                    return False
                next_version = int(current["version"]) + 1
                await conn.execute(
                    """UPDATE agent_tasks SET status='completed', result=$2::jsonb,
                       completed_at=NOW(), completed_by=$3, owner=NULL,
                       execution_token=NULL, lease_until=NULL, version=$4,
                       updated_at=NOW() WHERE id=$1""",
                    task_id, _json(result), worker_id, next_version,
                )
                await self._event(
                    conn, task_id, current["tenant_id"], current["user_id"],
                    next_version, "completed", "running", "completed", {},
                )
                return True

    async def fail(
        self, task_id: str, worker_id: str, token: str, error: str,
        topic: str, retry_delay_seconds: int,
    ) -> str:
        async with db_manager.get_connection() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    "SELECT * FROM agent_tasks WHERE id=$1 FOR UPDATE", task_id
                )
                if not row:
                    return "stale"
                current = dict(row)
                if (
                    current["status"] == "cancelling"
                    and current.get("owner") == worker_id
                    and current.get("execution_token") == token
                ):
                    next_version = int(current["version"]) + 1
                    await conn.execute(
                        """UPDATE agent_tasks SET status='cancelled',
                           completed_at=NOW(), owner=NULL, execution_token=NULL,
                           lease_until=NULL, version=$2, updated_at=NOW()
                           WHERE id=$1""",
                        task_id, next_version,
                    )
                    await self._event(
                        conn, task_id, current["tenant_id"], current["user_id"],
                        next_version, "cancelled", "cancelling", "cancelled", {},
                    )
                    return "cancelled"
                if (
                    current["status"] != "running"
                    or current.get("owner") != worker_id
                    or current.get("execution_token") != token
                ):
                    return "stale"
                retry_count = int(current["retry_count"]) + 1
                next_version = int(current["version"]) + 1
                if retry_count <= int(current["max_retries"]):
                    available_at = datetime.now(timezone.utc) + timedelta(
                        seconds=retry_delay_seconds
                    )
                    await conn.execute(
                        """UPDATE agent_tasks SET status='retry_wait', retry_count=$2,
                           error=$3, next_run_at=$4, owner=NULL, execution_token=NULL,
                           lease_until=NULL, version=$5, updated_at=NOW() WHERE id=$1""",
                        task_id, retry_count, error, available_at, next_version,
                    )
                    await self._event(
                        conn, task_id, current["tenant_id"], current["user_id"],
                        next_version, "retry_scheduled", "running", "retry_wait",
                        {"retry_count": retry_count, "error": error},
                    )
                    await self._outbox(
                        conn, task_id, f"{task_id}:dispatch:{next_version}", topic,
                        {"task_id": task_id, "version": next_version}, available_at,
                    )
                    return "retry"
                await conn.execute(
                    """UPDATE agent_tasks SET status='failed', retry_count=$2,
                       error=$3, completed_at=NOW(), owner=NULL,
                       execution_token=NULL, lease_until=NULL, version=$4,
                       updated_at=NOW() WHERE id=$1""",
                    task_id, retry_count, error, next_version,
                )
                await self._event(
                    conn, task_id, current["tenant_id"], current["user_id"],
                    next_version, "failed", "running", "failed",
                    {"retry_count": retry_count, "error": error},
                )
                return "failed"

    async def cancel(self, task_id: str) -> bool:
        async with db_manager.get_connection() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    "SELECT * FROM agent_tasks WHERE id=$1 FOR UPDATE", task_id
                )
                if not row or row["status"] in TERMINAL_STATUSES:
                    return False
                old = row["status"]
                target = "cancelling" if old == "running" else "cancelled"
                next_version = int(row["version"]) + 1
                await conn.execute(
                    """UPDATE agent_tasks SET status=$2, cancel_requested_at=NOW(),
                       completed_at=CASE WHEN $2='cancelled' THEN NOW() ELSE completed_at END,
                       version=$3, updated_at=NOW() WHERE id=$1""",
                    task_id, target, next_version,
                )
                await self._event(
                    conn, task_id, row["tenant_id"], row["user_id"], next_version,
                    "cancel_requested", old, target, {},
                )
                return True

    async def get(self, task_id: str, include_events: bool = True) -> Optional[Dict[str, Any]]:
        async with db_manager.get_connection() as conn:
            row = await conn.fetchrow("SELECT * FROM agent_tasks WHERE id=$1", task_id)
            task = self._row(row)
            if task and include_events:
                events = await conn.fetch(
                    """SELECT event_type, from_status, to_status, payload, version,
                              created_at
                       FROM agent_task_events WHERE task_id=$1 ORDER BY version""",
                    task_id,
                )
                task["events"] = [
                    {
                        "type": event["event_type"],
                        "from_status": event["from_status"],
                        "to_status": event["to_status"],
                        "data": _decode_json(event["payload"], {}),
                        "version": event["version"],
                        "time": _iso(event["created_at"]),
                    }
                    for event in events
                ]
            return task

    async def list(
        self, limit: int, tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        clauses = []
        values: List[Any] = []
        if tenant_id is not None:
            values.append(tenant_id)
            clauses.append(f"tenant_id=${len(values)}")
        if user_id is not None:
            values.append(user_id)
            clauses.append(f"user_id=${len(values)}")
        values.append(limit)
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        async with db_manager.get_connection() as conn:
            rows = await conn.fetch(
                f"SELECT * FROM agent_tasks {where} ORDER BY created_at DESC LIMIT ${len(values)}",
                *values,
            )
            return [self._row(row) for row in rows]

    async def append_event(self, task_id: str, event: Dict[str, Any]) -> None:
        async with db_manager.get_connection() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """UPDATE agent_tasks SET version=version+1, updated_at=NOW()
                       WHERE id=$1 RETURNING tenant_id,user_id,status,version""",
                    task_id,
                )
                if row:
                    await self._event(
                        conn, task_id, row["tenant_id"], row["user_id"], row["version"],
                        event.get("type", "progress"), row["status"], row["status"], event,
                    )

    async def queue_size(self) -> int:
        async with db_manager.get_connection() as conn:
            return int(await conn.fetchval(
                """SELECT COUNT(*) FROM agent_tasks
                   WHERE status IN ('pending','retry_wait')"""
            ))

    @staticmethod
    async def _event(
        conn: Any, task_id: str, tenant_id: str, user_id: str, version: int,
        event_type: str, from_status: Optional[str], to_status: Optional[str],
        payload: Dict[str, Any],
    ) -> None:
        await conn.execute(
            """INSERT INTO agent_task_events
               (task_id,tenant_id,user_id,version,event_type,from_status,to_status,payload)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8::jsonb)""",
            task_id, tenant_id, user_id, version, event_type, from_status,
            to_status, _json(payload),
        )

    @staticmethod
    async def _outbox(
        conn: Any, task_id: str, event_key: str, topic: str,
        payload: Dict[str, Any], available_at: datetime,
    ) -> None:
        await conn.execute(
            """INSERT INTO agent_task_outbox
               (task_id,event_key,topic,payload,available_at)
               VALUES ($1,$2,$3,$4::jsonb,$5)
               ON CONFLICT(event_key) DO NOTHING""",
            task_id, event_key, topic, _json(payload), available_at,
        )


postgres_task_store = PostgresTaskStore()
