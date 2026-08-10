"""
InnoCore AI vector store manager.

The public manager API is backend-neutral. Qdrant and pgvector are selected
through VECTOR_DB_TYPE without leaking backend-specific filters into agents.
"""

import asyncio
import logging
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .config import VectorDBType, get_config
from .exceptions import VectorStoreException

logger = logging.getLogger(__name__)


class LangChainEmbeddings(Embeddings):
    """Adapt the project's async embedding service to LangChain."""

    def __init__(self, embedding_service):
        self.embedding_service = embedding_service
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="embedding_"
        )

    def _run_async_in_thread(self, coroutine):
        def run_in_new_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(coroutine)
            finally:
                loop.close()

        return self._executor.submit(run_in_new_loop).result(timeout=120)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.embedding_service.generate_batch_embeddings(texts)
            )
        return self._run_async_in_thread(
            self.embedding_service.generate_batch_embeddings(texts)
        )

    def embed_query(self, text: str) -> List[float]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.embedding_service.generate_embedding(text))
        return self._run_async_in_thread(
            self.embedding_service.generate_embedding(text)
        )

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await self.embedding_service.generate_batch_embeddings(texts)

    async def aembed_query(self, text: str) -> List[float]:
        return await self.embedding_service.generate_embedding(text)

    def close(self) -> None:
        self._executor.shutdown(wait=False)


class VectorStoreManager:
    """Backend-neutral L1/L2 vector store manager."""

    def __init__(self):
        app_config = get_config()
        self.config = app_config.vector_db
        self.database_config = app_config.database
        self.backend = self.config.db_type
        self.client = None
        self.l1_collection = f"{self.config.collection_name_prefix}_l1_preset"
        self.l2_collection = f"{self.config.collection_name_prefix}_l2_user"
        self.l1_vectorstore = None
        self.l2_vectorstore = None
        self.embeddings: Optional[LangChainEmbeddings] = None
        self.embedding_dimension: Optional[int] = None
        self._pg_connection_string: Optional[str] = None

    async def initialize(self, embedding_service=None):
        """Initialize the selected vector database and its collections."""
        if embedding_service is None:
            raise VectorStoreException(
                "初始化向量数据库必须提供 embedding_service"
            )

        try:
            self.embeddings = LangChainEmbeddings(embedding_service)
            self.embedding_dimension = await self._get_embedding_dimension()

            if self.backend == VectorDBType.QDRANT:
                await self._initialize_qdrant()
            elif self.backend == VectorDBType.PGVECTOR:
                await self._initialize_pgvector()
            else:
                raise VectorStoreException(
                    f"当前未实现向量数据库后端: {self.backend.value}"
                )
        except VectorStoreException:
            raise
        except Exception as exc:
            raise VectorStoreException(
                f"{self.backend.value} 初始化失败: {exc}"
            ) from exc

    async def _get_embedding_dimension(self) -> int:
        try:
            embedding = await self.embeddings.aembed_query("dimension probe")
        except Exception as exc:
            raise VectorStoreException(
                f"无法检测 Embedding 维度: {exc}"
            ) from exc
        if not embedding:
            raise VectorStoreException("Embedding 服务返回了空向量")
        return len(embedding)

    async def _initialize_qdrant(self) -> None:
        try:
            from langchain_qdrant import QdrantVectorStore
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError as exc:
            raise VectorStoreException(
                "使用 Qdrant 需要安装 langchain-qdrant 和 qdrant-client"
            ) from exc

        self.client = QdrantClient(
            host=self.config.host,
            port=self.config.port,
            api_key=self.config.api_key,
            prefer_grpc=False,
            https=self.config.https,
            check_compatibility=False,
        )

        for collection_name in (self.l1_collection, self.l2_collection):
            if self.client.collection_exists(collection_name):
                info = self.client.get_collection(collection_name)
                existing_dimension = info.config.params.vectors.size
                if existing_dimension != self.embedding_dimension:
                    if not self.config.recreate_on_dimension_mismatch:
                        raise VectorStoreException(
                            f"Qdrant collection {collection_name} 的维度为 "
                            f"{existing_dimension}，当前 Embedding 维度为 "
                            f"{self.embedding_dimension}。请迁移/重建索引，或在确认"
                            "可丢弃旧向量后设置 "
                            "VECTOR_DB_RECREATE_ON_DIMENSION_MISMATCH=true"
                        )
                    logger.warning("重建维度不匹配的 collection: %s", collection_name)
                    self.client.delete_collection(collection_name)
                    self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(
                            size=self.embedding_dimension,
                            distance=Distance.COSINE,
                        ),
                    )
            else:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dimension,
                        distance=Distance.COSINE,
                    ),
                )

        self.l1_vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.l1_collection,
            embedding=self.embeddings,
        )
        self.l2_vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.l2_collection,
            embedding=self.embeddings,
        )

    def _build_pg_connection_string(self) -> str:
        configured = self.config.pgvector_connection_string
        if configured:
            if configured.startswith("postgresql://"):
                return configured.replace(
                    "postgresql://", "postgresql+psycopg://", 1
                )
            if configured.startswith("postgres://"):
                return configured.replace(
                    "postgres://", "postgresql+psycopg://", 1
                )
            return configured

        db = self.database_config
        return (
            "postgresql+psycopg://"
            f"{quote_plus(db.username)}:{quote_plus(db.password)}"
            f"@{db.host}:{db.port}/{db.database}"
        )

    def _psycopg_connection_string(self) -> str:
        return self._pg_connection_string.replace(
            "postgresql+psycopg://", "postgresql://", 1
        )

    async def _initialize_pgvector(self) -> None:
        try:
            from langchain_postgres import PGVector
        except ImportError as exc:
            raise VectorStoreException(
                "使用 pgvector 需要安装 langchain-postgres 和 psycopg"
            ) from exc

        self._pg_connection_string = self._build_pg_connection_string()
        await asyncio.to_thread(self._ensure_pgvector_extension)

        common_options = {
            "embeddings": self.embeddings,
            "connection": self._pg_connection_string,
            "use_jsonb": True,
            "embedding_length": self.embedding_dimension,
            "create_extension": False,
        }
        self.l1_vectorstore = PGVector(
            collection_name=self.l1_collection,
            collection_metadata={
                "embedding_dimension": self.embedding_dimension,
                "embedding_model": self.config.embedding_model,
            },
            **common_options,
        )
        await asyncio.to_thread(self._validate_pgvector_dimension)
        self.l2_vectorstore = PGVector(
            collection_name=self.l2_collection,
            collection_metadata={
                "embedding_dimension": self.embedding_dimension,
                "embedding_model": self.config.embedding_model,
            },
            **common_options,
        )

    def _ensure_pgvector_extension(self) -> None:
        try:
            import psycopg
        except ImportError as exc:
            raise VectorStoreException(
                "pgvector 后端缺少 psycopg 依赖"
            ) from exc

        try:
            with psycopg.connect(self._psycopg_connection_string()) as connection:
                with connection.cursor() as cursor:
                    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                connection.commit()
        except Exception as exc:
            raise VectorStoreException(
                "无法启用 PostgreSQL vector 扩展。请使用带 pgvector 的镜像，"
                "并确保当前用户具有 CREATE EXTENSION 权限。"
            ) from exc

    def _validate_pgvector_dimension(self) -> None:
        import psycopg

        query = """
            SELECT format_type(attribute.atttypid, attribute.atttypmod)
            FROM pg_attribute AS attribute
            JOIN pg_class AS table_info
              ON table_info.oid = attribute.attrelid
            WHERE table_info.relname = 'langchain_pg_embedding'
              AND attribute.attname = 'embedding'
              AND NOT attribute.attisdropped
        """
        with psycopg.connect(self._psycopg_connection_string()) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
                row = cursor.fetchone()

        column_type = row[0] if row else ""
        match = re.fullmatch(r"vector\((\d+)\)", column_type)
        if match and int(match.group(1)) != self.embedding_dimension:
            raise VectorStoreException(
                "pgvector 表 langchain_pg_embedding 的维度为 "
                f"{match.group(1)}，当前 Embedding 维度为 "
                f"{self.embedding_dimension}。请使用新的数据库/schema，"
                "或在备份后重建 pgvector 表并重新生成全部向量。"
            )

    def _document_id(
        self, collection_name: str, paper_id: str, user_id: Optional[str] = None
    ) -> str:
        source = f"{collection_name}:{user_id or ''}:{paper_id}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, source))

    async def add_to_l1(
        self,
        paper_id: str,
        title: str,
        abstract: str,
        content: str,
        metadata: Dict = None,
    ) -> str:
        document = self._build_document(
            paper_id, title, abstract, content, "l1", metadata
        )
        return await self._add_document(
            self.l1_vectorstore,
            document,
            self._document_id(self.l1_collection, paper_id),
            "L1",
        )

    async def add_to_l2(
        self,
        user_id: str,
        paper_id: str,
        title: str,
        abstract: str,
        content: str,
        metadata: Dict = None,
    ) -> str:
        document = self._build_document(
            paper_id,
            title,
            abstract,
            content,
            "l2",
            {"user_id": user_id, **(metadata or {})},
        )
        return await self._add_document(
            self.l2_vectorstore,
            document,
            self._document_id(self.l2_collection, paper_id, user_id),
            "L2",
        )

    @staticmethod
    def _build_document(
        paper_id: str,
        title: str,
        abstract: str,
        content: str,
        collection_type: str,
        metadata: Optional[Dict],
    ) -> Document:
        return Document(
            page_content=f"{title} {abstract} {content}",
            metadata={
                "paper_id": paper_id,
                "title": title,
                "abstract": abstract,
                "collection_type": collection_type,
                **(metadata or {}),
            },
        )

    async def _add_document(
        self, vectorstore, document: Document, document_id: str, label: str
    ) -> str:
        if vectorstore is None:
            raise VectorStoreException(f"{label} 向量存储未初始化")
        try:
            ids = await asyncio.to_thread(
                vectorstore.add_documents, [document], ids=[document_id]
            )
            return str(ids[0]) if ids else ""
        except Exception as exc:
            raise VectorStoreException(f"添加到{label}库失败: {exc}") from exc

    def _user_filter(self, user_id: str):
        if self.backend == VectorDBType.PGVECTOR:
            return {"user_id": {"$eq": user_id}}

        from qdrant_client.models import FieldCondition, Filter, MatchValue

        return Filter(
            must=[
                FieldCondition(
                    key="metadata.user_id",
                    match=MatchValue(value=user_id),
                )
            ]
        )

    async def hybrid_search(
        self,
        query: str,
        user_id: str = None,
        top_k: int = 5,
        include_l1: bool = True,
        include_l2: bool = True,
    ) -> List[Dict]:
        """Search vectors and combine normalized relevance with keyword overlap."""
        try:
            app_config = get_config()
            vector_weight = app_config.hybrid_search_weights.get("vector", 0.7)
            keyword_weight = app_config.hybrid_search_weights.get("keyword", 0.3)
            results = []

            if include_l1 and self.l1_vectorstore:
                matches = await asyncio.to_thread(
                    self.l1_vectorstore.similarity_search_with_relevance_scores,
                    query,
                    k=top_k,
                )
                results.extend(
                    self._format_matches(matches, "l1", vector_weight)
                )

            if include_l2 and user_id and self.l2_vectorstore:
                matches = await asyncio.to_thread(
                    self.l2_vectorstore.similarity_search_with_relevance_scores,
                    query,
                    k=top_k,
                    filter=self._user_filter(user_id),
                )
                results.extend(
                    self._format_matches(matches, "l2", vector_weight)
                )

            for result in results:
                payload = result["payload"]
                keyword_score = self._calculate_keyword_score(
                    query,
                    f"{payload.get('title', '')} {payload.get('abstract', '')}",
                )
                result["score"] += keyword_score * keyword_weight

            results.sort(key=lambda item: item["score"], reverse=True)
            return results[:top_k]
        except Exception as exc:
            raise VectorStoreException(f"混合搜索失败: {exc}") from exc

    @staticmethod
    def _format_matches(matches, collection_type: str, vector_weight: float):
        formatted = []
        for document, relevance in matches:
            metadata = document.metadata
            formatted.append(
                {
                    "id": metadata.get("paper_id", ""),
                    "score": max(0.0, min(1.0, float(relevance))) * vector_weight,
                    "payload": {
                        "paper_id": metadata.get("paper_id", ""),
                        "title": metadata.get("title", ""),
                        "abstract": metadata.get("abstract", ""),
                        **metadata,
                    },
                    "collection_type": collection_type,
                }
            )
        return formatted

    @staticmethod
    def _calculate_keyword_score(query: str, content: str) -> float:
        query_words = set(query.lower().split())
        if not query_words:
            return 0.0
        return len(query_words.intersection(content.lower().split())) / len(
            query_words
        )

    async def get_user_vectors(
        self, user_id: str, limit: int = 100
    ) -> List[Dict]:
        if self.backend == VectorDBType.PGVECTOR:
            rows = await asyncio.to_thread(
                self._pg_fetch_user_vectors, user_id, limit
            )
            return [{"id": str(row[0]), "payload": row[1]} for row in rows]

        results = await asyncio.to_thread(
            self.client.scroll,
            collection_name=self.l2_collection,
            scroll_filter=self._user_filter(user_id),
            limit=limit,
            with_payload=True,
        )
        return [
            {"id": str(point.id), "payload": point.payload}
            for point in results[0]
        ]

    def _pg_fetch_user_vectors(self, user_id: str, limit: int):
        import psycopg

        query = """
            SELECT embedding.id, embedding.cmetadata
            FROM langchain_pg_embedding AS embedding
            JOIN langchain_pg_collection AS collection
              ON collection.uuid = embedding.collection_id
            WHERE collection.name = %s
              AND embedding.cmetadata ->> 'user_id' = %s
            ORDER BY embedding.id
            LIMIT %s
        """
        with psycopg.connect(self._psycopg_connection_string()) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (self.l2_collection, user_id, limit))
                return cursor.fetchall()

    async def delete_user_vectors(self, user_id: str) -> bool:
        if self.backend == VectorDBType.PGVECTOR:
            await asyncio.to_thread(self._pg_delete_user_vectors, user_id)
            return True

        from qdrant_client.models import FilterSelector

        await asyncio.to_thread(
            self.client.delete,
            collection_name=self.l2_collection,
            points_selector=FilterSelector(
                filter=self._user_filter(user_id)
            ),
        )
        return True

    def _pg_delete_user_vectors(self, user_id: str) -> None:
        import psycopg

        query = """
            DELETE FROM langchain_pg_embedding AS embedding
            USING langchain_pg_collection AS collection
            WHERE collection.uuid = embedding.collection_id
              AND collection.name = %s
              AND embedding.cmetadata ->> 'user_id' = %s
        """
        with psycopg.connect(self._psycopg_connection_string()) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (self.l2_collection, user_id))
            connection.commit()

    async def get_collection_info(self, collection_type: str = "l1"):
        collection_name = (
            self.l1_collection if collection_type == "l1" else self.l2_collection
        )
        if self.backend == VectorDBType.QDRANT:
            return await asyncio.to_thread(
                self.client.get_collection, collection_name
            )
        count = await asyncio.to_thread(
            self._pg_collection_count, collection_name
        )
        return {
            "backend": "pgvector",
            "collection_name": collection_name,
            "points_count": count,
            "embedding_dimension": self.embedding_dimension,
        }

    def _pg_collection_count(self, collection_name: str) -> int:
        import psycopg

        query = """
            SELECT count(*)
            FROM langchain_pg_embedding AS embedding
            JOIN langchain_pg_collection AS collection
              ON collection.uuid = embedding.collection_id
            WHERE collection.name = %s
        """
        with psycopg.connect(self._psycopg_connection_string()) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (collection_name,))
                return cursor.fetchone()[0]

    def get_retriever(
        self, collection_type: str = "l1", search_kwargs: Dict = None
    ):
        vectorstore = (
            self.l1_vectorstore if collection_type == "l1" else self.l2_vectorstore
        )
        if not vectorstore:
            raise VectorStoreException(
                f"{collection_type} 向量存储未初始化"
            )
        return vectorstore.as_retriever(
            search_kwargs=search_kwargs or {"k": 5}
        )

    async def close(self):
        if self.client:
            await asyncio.to_thread(self.client.close)
        for vectorstore in (self.l1_vectorstore, self.l2_vectorstore):
            engine = getattr(vectorstore, "_engine", None)
            if engine is not None:
                await asyncio.to_thread(engine.dispose)
        if self.embeddings:
            self.embeddings.close()

    def is_embedding_initialized(self) -> bool:
        return self.embeddings is not None

    def get_initialization_status(self) -> Dict[str, Any]:
        return {
            "backend": self.backend.value,
            "client_ready": self.client is not None
            if self.backend == VectorDBType.QDRANT
            else self.l1_vectorstore is not None,
            "qdrant_client_ready": (
                self.client is not None
                if self.backend == VectorDBType.QDRANT
                else False
            ),
            "l1_vectorstore_ready": self.l1_vectorstore is not None,
            "l2_vectorstore_ready": self.l2_vectorstore is not None,
            "embedding_service_ready": self.embeddings is not None,
            "embedding_dimension": self.embedding_dimension,
        }


vector_store_manager = VectorStoreManager()
