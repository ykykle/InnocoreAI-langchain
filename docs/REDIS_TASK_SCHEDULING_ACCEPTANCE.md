# Redis 多实例任务调度验收手册

## 1. 验收目标

验证以下行为：

1. 默认 `local` 模式不依赖 Redis，单实例功能保持正常。
2. `redis` 模式下所有实例共享任务状态，同一任务只能被一个 Worker 认领。
3. Worker 异常退出后，任务租约到期可被其他 Worker 恢复。
4. 同步业务接口保持原响应方式，异步任务接口继续返回任务 ID。
5. `/api/v1/agent/runs` 使用统一任务后端，不再依赖单进程内存。
6. Redis 不可用时，`redis` 模式启动失败，不静默降级。

## 2. 调度模式

### 单实例模式

```env
TASK_QUEUE_BACKEND=local
TASK_WORKER_ENABLED=true
```

任务存储在当前进程内。该模式不需要 Redis，适合本地开发。

### 多实例模式

```env
TASK_QUEUE_BACKEND=redis
TASK_WORKER_ENABLED=true
TASK_QUEUE_KEY_PREFIX=innocore:acceptance
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
```

Redis 是唯一任务源。每个实例可以同时提供 API 和消费任务。

调度语义是 **at-least-once**：租约和 fencing token 保证同一时刻只有一个
有效持有者可以提交最终结果，但 Worker 在外部副作用完成后、提交结果前崩溃时，
任务仍可能重试。因此写数据库、发消息等有副作用的 Agent 工具仍需按 `task_id`
实现幂等。

如果需要拆分 API 与 Worker：

```env
# API 实例
TASK_QUEUE_BACKEND=redis
TASK_WORKER_ENABLED=false

# Worker 实例
TASK_QUEUE_BACKEND=redis
TASK_WORKER_ENABLED=true
```

项目目前通过 FastAPI 生命周期启动 Worker，因此至少要有一个
`TASK_WORKER_ENABLED=true` 的应用实例。

## 3. 自动化测试

### 3.1 单实例后端

```bash
python -m unittest tests.test_task_queue -v
```

预期：8 项测试全部通过，覆盖模式配置、优先级、唯一认领、fencing token、
重试、取消、事件和结果读取。

### 3.2 Redis 集成测试

先启动专用测试 Redis，禁止使用包含生产数据的实例：

```bash
docker run --rm -d \
  --name innocore-redis-acceptance \
  -p 6389:6379 \
  redis:7-alpine \
  redis-server --save "" --appendonly no
```

运行：

```bash
RUN_REDIS_INTEGRATION=1 \
REDIS_TEST_PORT=6389 \
python -m unittest tests.test_redis_task_queue_integration -v
```

预期：5 项测试全部通过。测试使用随机 `innocore:test:*` 前缀并在结束时清理。

```bash
docker stop innocore-redis-acceptance
```

## 4. 单实例 API 验收

启动：

```bash
TASK_QUEUE_BACKEND=local \
TASK_WORKER_ENABLED=true \
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

检查调度状态：

```bash
curl -s http://127.0.0.1:8000/health
```

确认：

```text
components.task_queue.backend == "local"
components.task_queue.redis_available == false
components.task_queue.worker_enabled == true
```

提交一个不依赖外部检索的引用任务：

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/tasks/submit \
  -H 'Content-Type: application/json' \
  -d '{
    "task_type": "citation_validation",
    "priority": 5,
    "input_data": {
      "paper_info": {
        "title": "Distributed Task Scheduling",
        "authors": ["Test Author"],
        "year": 2026
      },
      "formats": ["bibtex"],
      "verify_external": false
    }
  }'
```

记录返回的 `task_id`，然后查询：

```bash
curl -s http://127.0.0.1:8000/api/v1/tasks/TASK_ID/status
curl -s http://127.0.0.1:8000/api/v1/tasks/TASK_ID/execute
```

确认最终状态为 `completed`，并能获取 BibTeX 结果。

## 5. 多实例 API 验收

先启动 Redis：

```bash
docker compose up -d redis
docker compose ps redis
```

分别启动两个实例：

```bash
TASK_QUEUE_BACKEND=redis \
TASK_QUEUE_KEY_PREFIX=innocore:acceptance \
INSTANCE_ID=api-1 \
uvicorn api.main:app --host 127.0.0.1 --port 8001
```

```bash
TASK_QUEUE_BACKEND=redis \
TASK_QUEUE_KEY_PREFIX=innocore:acceptance \
INSTANCE_ID=api-2 \
uvicorn api.main:app --host 127.0.0.1 --port 8002
```

向实例 1 提交任务，然后从实例 2 查询：

```bash
curl -s -X POST http://127.0.0.1:8001/api/v1/tasks/submit \
  -H 'Content-Type: application/json' \
  -d '{
    "task_type": "citation_validation",
    "priority": 9,
    "input_data": {
      "paper_info": {
        "title": "Redis Lease Queue",
        "authors": ["Acceptance User"],
        "year": 2026
      },
      "formats": ["bibtex"],
      "verify_external": false
    }
  }'
```

```bash
curl -s http://127.0.0.1:8002/api/v1/tasks/TASK_ID/status
```

确认：

- 实例 2 能查询到实例 1 创建的任务。
- 最终状态为 `completed`。
- `completed_by` 只有一个 Worker ID。
- `retry_count` 为 `0`。

检查 Redis：

```bash
redis-cli ZCARD innocore:acceptance:tasks:pending
redis-cli ZCARD innocore:acceptance:tasks:processing
redis-cli XRANGE innocore:acceptance:tasks:history - +
redis-cli HGETALL innocore:acceptance:task:TASK_ID
```

任务完成后，pending 和 processing 均应不包含该任务；history 中只有一条
对应的 terminal 事件。

## 6. Worker 故障恢复

1. 提交一个执行时间超过 30 秒的任务。
2. 从任务状态的 `owner` 确认执行 Worker。
3. 强制停止该 Worker。
4. 等待 `TASK_LEASE_SECONDS`。
5. 保持另一个 Worker 运行并查询任务状态。

预期：

- 任务被重新放入 pending。
- `retry_count` 增加 1。
- `owner` 变为新的 Worker。
- 旧 Worker 即使恢复，也因 lease token 不匹配而无法提交结果。

测试时可临时缩短：

```env
TASK_LEASE_SECONDS=10
TASK_HEARTBEAT_SECONDS=2
```

## 7. Fail-fast 验收

停止 Redis 后执行：

```bash
TASK_QUEUE_BACKEND=redis \
REDIS_PORT=6399 \
uvicorn api.main:app --host 127.0.0.1 --port 8010
```

预期：应用启动失败，并记录 Redis 连接错误。

再执行：

```bash
TASK_QUEUE_BACKEND=local \
REDIS_PORT=6399 \
uvicorn api.main:app --host 127.0.0.1 --port 8010
```

预期：应用正常启动，健康检查显示 `task_queue.backend=local`。

## 8. 上线检查

- 所有实例使用相同 `TASK_QUEUE_KEY_PREFIX`。
- 不同环境使用不同前缀。
- 至少一个实例启用 Worker。
- `TASK_HEARTBEAT_SECONDS < TASK_LEASE_SECONDS`。
- Redis 使用 `noeviction`，并启用 AOF `everysec`。
- 监控 pending 数量、最老任务年龄、processing 数量、重试数、失败数、
  Redis 内存和命令错误。
- 确认没有旧版无前缀的 `task_queue`、`active_tasks`、`task_history` 持续增长。
