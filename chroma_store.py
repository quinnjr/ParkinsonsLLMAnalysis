"""
ChromaDB store for pipeline artifact RAG.

Author: Joseph R. Quinn <quinn.josephr@protonmail.com>
License: MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from artifact_ingest import WORK_TYPE_TO_COLLECTION, DocumentRecord


def build_store(docs: list[DocumentRecord], db_path: str | Path) -> None:
    """Build ChromaDB collections grouped by work_type."""
    import chromadb

    path = Path(db_path)
    path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(path))

    by_collection: dict[str, list[DocumentRecord]] = {}
    for doc in docs:
        by_collection.setdefault(doc.collection, []).append(doc)

    for collection_name, records in by_collection.items():
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
        collection = client.get_or_create_collection(name=collection_name)
        if not records:
            continue
        collection.add(
            ids=[r.doc_id for r in records],
            documents=[r.summary_text for r in records],
            metadatas=[
                {
                    "subject_id": r.subject_id,
                    "work_type": r.work_type,
                    "source_file": r.source_file,
                    "raw_payload": r.raw_payload[:2000],
                }
                for r in records
            ],
        )


def _get_client(db_path: str | Path):
    import chromadb

    return chromadb.PersistentClient(path=str(db_path))


def query_subject(
    db_path: str | Path,
    subject_id: str,
    work_types: list[str],
    query_text: str,
    k: int = 5,
) -> list[dict[str, Any]]:
    """Query collections for subject-specific or cohort-level context."""
    client = _get_client(db_path)
    chunks: list[dict[str, Any]] = []

    for work_type in work_types:
        collection_name = WORK_TYPE_TO_COLLECTION.get(work_type, work_type)
        try:
            collection = client.get_collection(collection_name)
        except Exception:
            continue

        n = min(k * 3, max(collection.count(), 1))
        result = collection.query(
            query_texts=[query_text],
            n_results=n,
        )

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        for idx, doc_id in enumerate(ids):
            meta = metadatas[idx] if idx < len(metadatas) else {}
            sid = meta.get("subject_id", "")
            if sid not in {subject_id, "COHORT"}:
                continue
            chunks.append(
                {
                    "doc_id": doc_id,
                    "text": documents[idx] if idx < len(documents) else "",
                    "metadata": meta,
                    "distance": distances[idx] if idx < len(distances) else None,
                    "work_type": work_type,
                }
            )

    chunks.sort(key=lambda c: c.get("distance") or 999.0)
    return chunks[:k]


def format_rag_context(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks as prompt context."""
    if not chunks:
        return "No pipeline context retrieved."
    lines = ["Retrieved pipeline context:"]
    for i, chunk in enumerate(chunks, 1):
        wt = chunk.get("work_type", "unknown")
        text = chunk.get("text", "")
        lines.append(f"{i}. [{wt}] {text}")
    return "\n".join(lines)
