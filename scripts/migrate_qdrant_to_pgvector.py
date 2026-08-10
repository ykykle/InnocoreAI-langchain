#!/usr/bin/env python3
"""Copy InnoCore L1/L2 vectors from Qdrant to pgvector."""

import argparse
import os
import uuid
from typing import List

from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector
from qdrant_client import QdrantClient


class PrecomputedEmbeddings(Embeddings):
    """PGVector requires an Embeddings object; migration only writes vectors."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise RuntimeError("迁移使用已有向量，不应重新生成 Embedding")

    def embed_query(self, text: str) -> List[float]:
        raise RuntimeError("迁移使用已有向量，不应执行查询")


def parse_args():
    parser = argparse.ArgumentParser(
        description="将 InnoCore 的 Qdrant collection 复制到 pgvector"
    )
    parser.add_argument(
        "--qdrant-host", default=os.getenv("QDRANT_HOST", "localhost")
    )
    parser.add_argument(
        "--qdrant-port",
        type=int,
        default=int(os.getenv("QDRANT_PORT", "6333")),
    )
    parser.add_argument(
        "--qdrant-api-key", default=os.getenv("QDRANT_API_KEY")
    )
    parser.add_argument(
        "--pg-connection",
        default=os.getenv("PGVECTOR_CONNECTION_STRING"),
        required=os.getenv("PGVECTOR_CONNECTION_STRING") is None,
        help="例如 postgresql://user:password@localhost:5432/innocore_ai",
    )
    parser.add_argument(
        "--collection-prefix",
        default=os.getenv("VECTOR_COLLECTION_PREFIX", "innocore"),
    )
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument(
        "--pre-delete-collections",
        action="store_true",
        help="先删除 pgvector 中的同名 collection（破坏性操作）",
    )
    return parser.parse_args()


def normalize_pg_connection(connection: str) -> str:
    if connection.startswith("postgresql+psycopg://"):
        return connection
    if connection.startswith("postgresql://"):
        return connection.replace(
            "postgresql://", "postgresql+psycopg://", 1
        )
    if connection.startswith("postgres://"):
        return connection.replace(
            "postgres://", "postgresql+psycopg://", 1
        )
    raise ValueError("pgvector 连接串必须使用 postgresql://")


def normalize_id(point_id) -> str:
    try:
        return str(uuid.UUID(str(point_id)))
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"qdrant:{point_id}"))


def unpack_payload(payload):
    payload = payload or {}
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {
            key: value
            for key, value in payload.items()
            if key not in {"page_content", "content", "metadata"}
        }
    text = payload.get("page_content") or payload.get("content")
    if not text:
        text = " ".join(
            str(payload.get(key, "")) for key in ("title", "abstract")
        ).strip()
    return text, metadata


def migrate_collection(
    source,
    destination,
    collection_name: str,
    expected_dimension: int,
    batch_size: int,
) -> int:
    offset = None
    migrated = 0
    while True:
        points, offset = source.scroll(
            collection_name=collection_name,
            offset=offset,
            limit=batch_size,
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            break

        texts, vectors, metadatas, ids = [], [], [], []
        for point in points:
            vector = point.vector
            if isinstance(vector, dict):
                if len(vector) != 1:
                    raise ValueError(
                        f"{collection_name} 使用多命名向量，无法自动迁移"
                    )
                vector = next(iter(vector.values()))
            if len(vector) != expected_dimension:
                raise ValueError(
                    f"{collection_name}/{point.id} 维度不一致: "
                    f"{len(vector)} != {expected_dimension}"
                )
            text, metadata = unpack_payload(point.payload)
            texts.append(text)
            vectors.append(vector)
            metadatas.append(metadata)
            ids.append(normalize_id(point.id))

        destination.add_embeddings(
            texts=texts,
            embeddings=vectors,
            metadatas=metadatas,
            ids=ids,
        )
        migrated += len(points)
        print(f"{collection_name}: 已迁移 {migrated} 条")
        if offset is None:
            break
    return migrated


def main():
    args = parse_args()
    source = QdrantClient(
        host=args.qdrant_host,
        port=args.qdrant_port,
        api_key=args.qdrant_api_key,
        prefer_grpc=False,
    )
    connection = normalize_pg_connection(args.pg_connection)
    collections = [
        f"{args.collection_prefix}_l1_preset",
        f"{args.collection_prefix}_l2_user",
    ]

    dimensions = {}
    for collection_name in collections:
        info = source.get_collection(collection_name)
        dimensions[collection_name] = info.config.params.vectors.size
    if len(set(dimensions.values())) != 1:
        raise ValueError(f"L1/L2 向量维度不一致: {dimensions}")
    dimension = next(iter(dimensions.values()))

    destinations = {
        collection_name: PGVector(
            embeddings=PrecomputedEmbeddings(),
            collection_name=collection_name,
            connection=connection,
            embedding_length=dimension,
            use_jsonb=True,
            create_extension=True,
            pre_delete_collection=args.pre_delete_collections,
        )
        for collection_name in collections
    }

    totals = {
        collection_name: migrate_collection(
            source,
            destinations[collection_name],
            collection_name,
            dimension,
            args.batch_size,
        )
        for collection_name in collections
    }
    print(f"迁移完成，维度={dimension}，记录数={totals}")


if __name__ == "__main__":
    main()
