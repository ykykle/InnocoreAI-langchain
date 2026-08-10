import uuid

from scripts.migrate_qdrant_to_pgvector import (
    normalize_id,
    normalize_pg_connection,
    unpack_payload,
)


def test_unpack_langchain_qdrant_payload():
    text, metadata = unpack_payload(
        {
            "page_content": "paper body",
            "metadata": {"paper_id": "p1", "user_id": "u1"},
        }
    )

    assert text == "paper body"
    assert metadata == {"paper_id": "p1", "user_id": "u1"}


def test_unpack_legacy_direct_payload():
    text, metadata = unpack_payload(
        {
            "title": "Title",
            "abstract": "Abstract",
            "content": "legacy body",
            "paper_id": "p1",
        }
    )

    assert text == "legacy body"
    assert metadata == {"title": "Title", "abstract": "Abstract", "paper_id": "p1"}


def test_connection_and_id_normalization_are_stable():
    assert normalize_pg_connection("postgresql://u:p@db/vectors") == (
        "postgresql+psycopg://u:p@db/vectors"
    )
    generated = normalize_id("legacy-qdrant-id")
    assert generated == normalize_id("legacy-qdrant-id")
    assert str(uuid.UUID(generated)) == generated
