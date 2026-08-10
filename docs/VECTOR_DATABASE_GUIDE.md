# 向量数据库配置与迁移

## 结论

本项目可以从 Qdrant 切换到 pgvector。当前业务只依赖以下能力：

- 余弦相似度检索
- JSON 元数据与 `user_id` 等值过滤
- L1 全局库和 L2 用户库
- 文档写入、用户向量查询和删除

这些能力 pgvector 均可覆盖。项目没有使用 Qdrant 的分片、副本、量化、稀疏向量或分布式集群能力，因此切换不需要修改 Agent 业务代码。

数据量较小、希望减少独立服务时，优先使用 pgvector；数据量很大、需要独立扩缩容、高并发向量检索或 Qdrant 高级能力时，保留 Qdrant。

## 配置

### Qdrant

```dotenv
VECTOR_DB_TYPE=qdrant
VECTOR_COLLECTION_PREFIX=innocore
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_HTTPS=false
# QDRANT_API_KEY=
```

### pgvector

`docker-compose.yml` 已使用 `pgvector/pgvector:pg16`，关系数据与向量数据可以放在同一个 PostgreSQL 实例中：

```dotenv
VECTOR_DB_TYPE=pgvector
VECTOR_COLLECTION_PREFIX=innocore

POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=innocore_ai
POSTGRES_USER=user
POSTGRES_PASSWORD=password
```

默认复用 `POSTGRES_*`。需要使用独立的向量数据库时设置：

```dotenv
PGVECTOR_CONNECTION_STRING=postgresql://user:password@host:5432/vector_db
```

数据库账号首次启动时需要 `CREATE EXTENSION` 权限。也可以由管理员预先执行：

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Embedding 凭据与向量数据库凭据相互独立：

```dotenv
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_API_KEY=...
```

## 从 Qdrant 迁移到 pgvector

1. 备份 Qdrant 和 PostgreSQL。
2. 安装新依赖并启动带 vector 扩展的 PostgreSQL。
3. 保持旧 Qdrant 可读，执行迁移脚本。
4. 对比源端和目标端 L1/L2 记录数，并抽样比较 Top-K 结果。
5. 将 `VECTOR_DB_TYPE` 改为 `pgvector`，重启应用并观察 `/health`。
6. 稳定运行一段时间后再停止 Qdrant；不要在验证前删除源数据。

迁移命令：

```bash
python3 scripts/migrate_qdrant_to_pgvector.py \
  --qdrant-host localhost \
  --qdrant-port 6333 \
  --pg-connection postgresql://user:password@localhost:5432/innocore_ai
```

脚本直接复制 Qdrant 中已有的向量和元数据，不会重新调用 Embedding API。重复迁移时使用相同 ID 更新记录。只有确认目标端旧 collection 可以删除时，才添加 `--pre-delete-collections`。

## 更换向量数据库时必须处理的事项

1. **依赖和基础设施**：安装对应 LangChain 驱动，准备数据库、持久化卷、备份和监控。
2. **连接配置**：设置后端类型、连接地址、TLS、凭据和 collection 前缀。
3. **Schema/collection**：创建 vector 扩展及 L1/L2 collection，确认元数据字段和用户过滤方式。
4. **距离与分数**：本项目统一使用余弦距离，并将结果转换为“分数越大越相关”。
5. **Embedding 兼容性**：迁移前后必须使用同一模型、维度和归一化策略。更换模型时应重新生成全部向量。
6. **数据迁移**：迁移正文、向量、文档 ID、`paper_id`、`user_id` 和其他元数据，并做数量校验。
7. **索引和性能**：大数据量下为 pgvector 建 HNSW/IVFFlat 索引并执行 `ANALYZE`；根据延迟调节连接池和检索参数。
8. **验证与回滚**：做写入、L2 用户隔离、删除、Top-K 召回和性能测试，切流期间保留 Qdrant 作为回滚源。

pgvector 数据量增长后可创建索引：

```sql
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_lc_pg_embedding_hnsw
ON langchain_pg_embedding
USING hnsw (embedding vector_cosine_ops);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_lc_pg_embedding_user
ON langchain_pg_embedding ((cmetadata ->> 'user_id'));

ANALYZE langchain_pg_embedding;
```

pgvector 的 `vector` 类型 HNSW 索引存在维度上限。当前默认模型的 1536 维可以直接使用；若改用超过上限的模型，需要选择降维、`halfvec`/其他索引方案或继续使用 Qdrant。

## 维度变化

应用启动时会探测实际 Embedding 维度。Qdrant collection 维度不一致时默认拒绝启动，不再自动删除数据。

仅在确认旧向量可以丢弃时设置：

```dotenv
VECTOR_DB_RECREATE_ON_DIMENSION_MISMATCH=true
```

生产环境更推荐使用新 collection 前缀完成全量重建和验证，再切换配置。
