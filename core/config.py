"""
InnoCore AI 核心配置模块
"""

from typing import Dict, Optional
from dataclasses import dataclass, field
from enum import Enum
import os
from dotenv import load_dotenv

load_dotenv()

class LLMProvider(Enum):
    """LLM提供商枚举"""
    OPENAI = "openai"
    CLAUDE = "claude"
    MODELSCOPE = "modelscope"  # 阿里云 ModelScope
    OLLAMA = "ollama"  # 本地部署
    DASHSCOPE = "dashscope"  # 阿里云灵积（推荐用于 Qwen 系列）

class VectorDBType(Enum):
    """向量数据库类型枚举"""
    QDRANT = "qdrant"
    PGVECTOR = "pgvector"
    CHROMA = "chroma"
    PINECONE = "pinecone"

@dataclass
class LLMConfig:
    """LLM配置"""
    provider: LLMProvider = LLMProvider.OPENAI
    model_name: str = "gpt-3.5-turbo"  # OpenAI: gpt-4, gpt-3.5-turbo, gpt-4-turbo-preview
                                        # DashScope: qwen-turbo, qwen-plus, qwen-max
                                        # ModelScope: qwen/Qwen2.5-7B-Instruct
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4000
    timeout: int = 60

@dataclass
class VectorDBConfig:
    """向量数据库配置"""
    db_type: VectorDBType = VectorDBType.QDRANT
    # Qdrant connection
    host: str = "localhost"
    port: int = 6333
    api_key: Optional[str] = None
    https: bool = False
    # pgvector connection. When omitted, DatabaseConfig is reused.
    pgvector_connection_string: Optional[str] = None
    collection_name_prefix: str = "innocore"
    recreate_on_dimension_mismatch: bool = False
    # Embedding service
    embedding_model: str = "text-embedding-3-small"
    embedding_base_url: Optional[str] = None
    embedding_api_key: Optional[str] = None
    embedding_provider: str = "openai"  # "openai" | "local"
    embedding_device: Optional[str] = None  # None=auto, "cpu", "cuda"

@dataclass
class DatabaseConfig:
    """关系数据库配置"""
    host: str = os.getenv("POSTGRES_HOST", "localhost")
    port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    database: str = os.getenv("POSTGRES_DB", "innocore_ai")
    username: str = os.getenv("POSTGRES_USER", "postgres")
    password: str = os.getenv("POSTGRES_PASSWORD", "password")
    pool_size: int = 10

@dataclass
class RedisConfig:
    """Redis配置"""
    url: Optional[str] = None
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_connections: int = 20
    socket_connect_timeout: float = 3.0
    socket_timeout: float = 5.0
    health_check_interval: int = 30


@dataclass
class TaskQueueConfig:
    """任务调度配置。local 用于开发，redis_stream 用于多实例生产。"""
    backend: str = "local"
    worker_enabled: bool = True
    instance_id: Optional[str] = None
    key_prefix: str = "innocore:dev"
    lease_seconds: int = 300
    heartbeat_seconds: int = 30
    max_retries: int = 3
    metadata_ttl: int = 86400
    result_ttl: int = 86400
    history_maxlen: int = 1000
    wait_timeout: int = 600
    poll_interval_ms: int = 200
    stream_name: str = "agent_tasks"
    consumer_group: str = "agent_workers"
    stream_maxlen: int = 10000
    stream_block_ms: int = 1000
    stream_claim_idle_ms: int = 300000
    outbox_batch_size: int = 100
    user_concurrency: int = 2
    miner_concurrency: int = 4

@dataclass
class ExternalAPIConfig:
    """外部API配置"""
    crossref_api_key: Optional[str] = None
    google_scholar_api_key: Optional[str] = None
    serpapi_key: Optional[str] = None
    arxiv_base_url: str = "http://export.arxiv.org/api/query"
    ieee_base_url: str = "https://ieeexploreapi.ieee.org/api/v1"
    ieee_api_key: Optional[str] = None

@dataclass
class InnoCoreConfig:
    """InnoCore AI 主配置类"""
    
    # 基础配置
    app_name: str = "InnoCore AI"
    debug: bool = False
    log_level: str = "INFO"
    
    # LLM配置
    llm: LLMConfig = field(default_factory=LLMConfig)
    
    # 向量数据库配置
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    
    # 关系数据库配置
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    
    # Redis配置
    redis: RedisConfig = field(default_factory=RedisConfig)

    # 任务调度配置
    task_queue: TaskQueueConfig = field(default_factory=TaskQueueConfig)
    
    # 外部API配置
    external_apis: ExternalAPIConfig = field(default_factory=ExternalAPIConfig)
    
    # Agent配置
    agent_max_steps: int = 5
    agent_timeout: int = 300
    concurrent_agents: int = 4
    
    # RAG配置
    retrieval_top_k: int = 5
    similarity_threshold: float = 0.7
    hybrid_search_weights: Dict[str, float] = field(default_factory=lambda: {
        "vector": 0.7,
        "keyword": 0.3
    })
    
    # 性能配置
    cache_ttl: int = 3600  # 缓存过期时间(秒)
    batch_size: int = 10
    max_concurrent_requests: int = 50
    
    def __post_init__(self):
        """初始化后处理"""
        # 从环境变量加载配置
        self.llm.api_key = self.llm.api_key or os.getenv("OPENAI_API_KEY")
        self.llm.base_url = self.llm.base_url or os.getenv("OPENAI_BASE_URL")
        
        # 从环境变量加载模型名称（如果设置了）
        env_model = os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL")
        if env_model:
            self.llm.model_name = env_model
        
        vector_db_type = os.getenv("VECTOR_DB_TYPE", self.vector_db.db_type.value).lower()
        try:
            self.vector_db.db_type = VectorDBType(vector_db_type)
        except ValueError as exc:
            supported = ", ".join(db_type.value for db_type in VectorDBType)
            raise ValueError(
                f"不支持的 VECTOR_DB_TYPE={vector_db_type!r}，可选值: {supported}"
            ) from exc

        self.vector_db.host = os.getenv("QDRANT_HOST", self.vector_db.host)
        self.vector_db.port = int(os.getenv("QDRANT_PORT", str(self.vector_db.port)))
        self.vector_db.api_key = os.getenv("QDRANT_API_KEY") or self.vector_db.api_key
        self.vector_db.https = os.getenv(
            "QDRANT_HTTPS", str(self.vector_db.https)
        ).lower() == "true"
        self.vector_db.pgvector_connection_string = (
            os.getenv("PGVECTOR_CONNECTION_STRING")
            or self.vector_db.pgvector_connection_string
        )
        self.vector_db.collection_name_prefix = os.getenv(
            "VECTOR_COLLECTION_PREFIX", self.vector_db.collection_name_prefix
        )
        self.vector_db.recreate_on_dimension_mismatch = os.getenv(
            "VECTOR_DB_RECREATE_ON_DIMENSION_MISMATCH",
            str(self.vector_db.recreate_on_dimension_mismatch),
        ).lower() == "true"

        # Embedding 配置与向量数据库凭据相互独立。
        embedding_model = os.getenv("EMBEDDING_MODEL")
        if embedding_model:
            self.vector_db.embedding_model = embedding_model
        
        embedding_base_url = os.getenv("EMBEDDING_BASE_URL")
        if embedding_base_url:
            self.vector_db.embedding_base_url = embedding_base_url
        embedding_provider = os.getenv("EMBEDDING_PROVIDER")
        if embedding_provider:
            self.vector_db.embedding_provider = embedding_provider
        embedding_device = os.getenv("EMBEDDING_DEVICE")
        if embedding_device:
            self.vector_db.embedding_device = embedding_device
        self.vector_db.embedding_api_key = (
            os.getenv("EMBEDDING_API_KEY") or self.vector_db.embedding_api_key
        )
        self.database.password = self.database.password or os.getenv("DATABASE_PASSWORD")
        self.redis.url = os.getenv("REDIS_URL") or self.redis.url
        self.redis.host = os.getenv("REDIS_HOST", self.redis.host)
        self.redis.port = int(os.getenv("REDIS_PORT", str(self.redis.port)))
        self.redis.db = int(os.getenv("REDIS_DB", str(self.redis.db)))
        self.redis.password = os.getenv("REDIS_PASSWORD") or self.redis.password
        self.redis.max_connections = int(os.getenv(
            "REDIS_MAX_CONNECTIONS", str(self.redis.max_connections)
        ))
        self.redis.socket_connect_timeout = float(os.getenv(
            "REDIS_CONNECT_TIMEOUT", str(self.redis.socket_connect_timeout)
        ))
        self.redis.socket_timeout = float(os.getenv(
            "REDIS_SOCKET_TIMEOUT", str(self.redis.socket_timeout)
        ))
        self.redis.health_check_interval = int(os.getenv(
            "REDIS_HEALTH_CHECK_INTERVAL", str(self.redis.health_check_interval)
        ))

        self.task_queue.backend = os.getenv(
            "TASK_QUEUE_BACKEND", self.task_queue.backend
        ).lower()
        # redis 作为旧配置名兼容，实际已切换为 PostgreSQL + Redis Stream。
        if self.task_queue.backend == "redis":
            self.task_queue.backend = "redis_stream"
        if self.task_queue.backend not in {"local", "redis_stream"}:
            raise ValueError("TASK_QUEUE_BACKEND 必须是 local 或 redis_stream")
        self.task_queue.worker_enabled = os.getenv(
            "TASK_WORKER_ENABLED", str(self.task_queue.worker_enabled)
        ).lower() == "true"
        self.task_queue.instance_id = (
            os.getenv("INSTANCE_ID") or self.task_queue.instance_id
        )
        self.task_queue.key_prefix = os.getenv(
            "TASK_QUEUE_KEY_PREFIX", self.task_queue.key_prefix
        )
        for env_name, attr in (
            ("TASK_LEASE_SECONDS", "lease_seconds"),
            ("TASK_HEARTBEAT_SECONDS", "heartbeat_seconds"),
            ("TASK_MAX_RETRIES", "max_retries"),
            ("TASK_METADATA_TTL", "metadata_ttl"),
            ("TASK_RESULT_TTL", "result_ttl"),
            ("TASK_HISTORY_MAXLEN", "history_maxlen"),
            ("TASK_WAIT_TIMEOUT", "wait_timeout"),
            ("TASK_POLL_INTERVAL_MS", "poll_interval_ms"),
            ("TASK_STREAM_MAXLEN", "stream_maxlen"),
            ("TASK_STREAM_BLOCK_MS", "stream_block_ms"),
            ("TASK_STREAM_CLAIM_IDLE_MS", "stream_claim_idle_ms"),
            ("TASK_OUTBOX_BATCH_SIZE", "outbox_batch_size"),
            ("TASK_USER_CONCURRENCY", "user_concurrency"),
            ("TASK_MINER_CONCURRENCY", "miner_concurrency"),
        ):
            setattr(
                self.task_queue,
                attr,
                int(os.getenv(env_name, str(getattr(self.task_queue, attr)))),
            )
        self.task_queue.stream_name = os.getenv(
            "TASK_STREAM_NAME", self.task_queue.stream_name
        )
        self.task_queue.consumer_group = os.getenv(
            "TASK_CONSUMER_GROUP", self.task_queue.consumer_group
        )
        if self.task_queue.heartbeat_seconds >= self.task_queue.lease_seconds:
            raise ValueError("TASK_HEARTBEAT_SECONDS 必须小于 TASK_LEASE_SECONDS")
        
        self.external_apis.crossref_api_key = self.external_apis.crossref_api_key or os.getenv("CROSSREF_API_KEY")
        self.external_apis.google_scholar_api_key = self.external_apis.google_scholar_api_key or os.getenv("GOOGLE_SCHOLAR_API_KEY")
        self.external_apis.serpapi_key = self.external_apis.serpapi_key or os.getenv("SERPAPI_KEY")
        self.external_apis.ieee_api_key = self.external_apis.ieee_api_key or os.getenv("IEEE_API_KEY")
        
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.agent_timeout = int(os.getenv("AGENT_TIMEOUT", str(self.agent_timeout)))
        self.agent_max_steps = int(os.getenv("AGENT_MAX_STEPS", str(self.agent_max_steps)))

# 全局配置实例
config = InnoCoreConfig()

def get_config() -> InnoCoreConfig:
    """获取全局配置实例"""
    return config

def update_config(**kwargs) -> None:
    """更新配置"""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
