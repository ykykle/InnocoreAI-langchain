from core.config import InnoCoreConfig, VectorDBType


def test_qdrant_and_embedding_credentials_are_independent(monkeypatch):
    monkeypatch.setenv("VECTOR_DB_TYPE", "qdrant")
    monkeypatch.setenv("QDRANT_API_KEY", "qdrant-secret")
    monkeypatch.setenv("EMBEDDING_API_KEY", "embedding-secret")

    config = InnoCoreConfig()

    assert config.vector_db.db_type == VectorDBType.QDRANT
    assert config.vector_db.api_key == "qdrant-secret"
    assert config.vector_db.embedding_api_key == "embedding-secret"


def test_pgvector_reuses_postgres_by_default(monkeypatch):
    monkeypatch.setenv("VECTOR_DB_TYPE", "pgvector")
    monkeypatch.delenv("PGVECTOR_CONNECTION_STRING", raising=False)

    config = InnoCoreConfig()

    assert config.vector_db.db_type == VectorDBType.PGVECTOR
    assert config.vector_db.pgvector_connection_string is None


def test_pgvector_accepts_independent_connection_string(monkeypatch):
    connection = "postgresql://vector_user:secret@db:5432/vectors"
    monkeypatch.setenv("VECTOR_DB_TYPE", "pgvector")
    monkeypatch.setenv("PGVECTOR_CONNECTION_STRING", connection)

    config = InnoCoreConfig()

    assert config.vector_db.pgvector_connection_string == connection
