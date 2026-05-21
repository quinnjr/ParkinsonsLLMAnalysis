import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from artifact_ingest import ingest_all
from chroma_store import build_store, query_subject

SYNTHETIC = Path(__file__).parent.parent / "example" / "synthetic"


@pytest.fixture
def chroma_db(tmp_path):
    params = {
        "syn_features": str(SYNTHETIC / "features.csv"),
        "stage1_json_dir": str(SYNTHETIC / "stage1"),
    }
    docs = ingest_all(params)
    db_path = tmp_path / "chroma"
    build_store(docs, db_path)
    return db_path


def test_build_store_creates_collections(chroma_db):
    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_db))
    names = {c.name for c in client.list_collections()}
    assert "proteomics" in names
    assert "stage1_outputs" in names


def test_query_subject_returns_chunks(chroma_db):
    chunks = query_subject(
        chroma_db, "PD_001", ["proteomics", "stage1"],
        "PD_001 proteomics alpha-synuclein", k=5,
    )
    assert len(chunks) >= 1
    assert any("PD_001" in c.get("text", "") or c.get("metadata", {}).get("subject_id") == "PD_001" for c in chunks)
