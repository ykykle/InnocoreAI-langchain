# 多用户并发安全任务执行

生产模式设置 `TASK_QUEUE_BACKEND=redis_stream`。该模式将职责明确拆开：

- PostgreSQL 是唯一状态真相源，保存任务、租约、执行令牌、结果、重试、取消和不可变审计事件。
- `agent_task_outbox` 与任务提交/重试处于同一个数据库事务，避免“数据库已提交但消息未发送”。
- Redis Stream 只传递 `task_id`，consumer group 保证一条投递在同一时刻由一个消费者持有；崩溃后的 pending 消息由 `XAUTOCLAIM` 接管。
- Worker 通过 PostgreSQL 行锁、版本号和 fencing token 认领、心跳及提交。旧 Worker 即使恢复，也不能覆盖新 Worker 的结果。

状态迁移如下：

```text
pending -> running -> completed
                   -> retry_wait -> running
                   -> failed
pending/retry_wait -> cancelled
running -> cancelling -> cancelled
```

准入采用两层限制：PostgreSQL 在认领顶层任务时执行跨实例原子计数；Agent 调用使用 Redis 租约信号量，覆盖完整工作流内部并发。默认每个 `(tenant_id, user_id)` 最多 2 个执行，全局 Miner 最多 4 个，可分别通过 `TASK_USER_CONCURRENCY` 和 `TASK_MINER_CONCURRENCY` 修改。

Agent 对象仍可作为单例保存工具和模型配置，但 history、state、LangGraph checkpointer 均按调用创建或由 `ContextVar` 隔离，不在用户之间共享。API 从可信认证网关注入的 `X-Tenant-ID`、`X-User-ID` 绑定任务归属；生产环境应设置 `AUTH_REQUIRED=true`，并在网关删除客户端伪造的同名请求头。

完整科研工作流使用以下失败策略：Hunter 失败立即使工作流失败；多个 Miner 采用 `BEST_EFFORT` 并保留成功结果；Validator 失败只写入 warnings；Coach 仅在至少存在一份有效 Miner 分析时执行。
