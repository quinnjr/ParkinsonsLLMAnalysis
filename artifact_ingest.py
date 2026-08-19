"""
Ingest Parkinson's pipeline artifacts into DocumentRecord objects.

Author: Joseph R. Quinn <quinn.josephr@protonmail.com>
License: MIT
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:  # available only inside the PluMA runtime
    import PyPluMA
except ImportError:  # pragma: no cover - exercised by standalone/test usage
    PyPluMA = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def pluma_prefix() -> str:
    """Return the PluMA pipeline prefix, or an empty string outside PluMA."""
    if PyPluMA is None:
        return ""
    try:
        return str(PyPluMA.prefix())
    except Exception:  # pragma: no cover - defensive
        return ""


def resolve_path(value: str | Path) -> str:
    """Resolve a configured path against the PluMA pipeline prefix.

    ``os.path.join`` leaves absolute values untouched, so absolute paths
    (e.g. ``work_dir /tmp/...``) and standalone/test usage keep working.
    """
    return os.path.join(pluma_prefix(), str(value))

WORK_TYPE_TO_COLLECTION: dict[str, str] = {
    "microbiome": "microbiome",
    "proteomics": "proteomics",
    "clinical_assay": "clinical_assay",
    "clinical": "clinical_labels",
    "integration_mofa": "integration_mofa",
    "integration_snf": "integration_snf",
    "integration_fusion": "integration_fusion",
    "explainability": "explainability",
    "ml_predictions": "ml_predictions",
    "stage1": "stage1_outputs",
}

PARAM_ARTIFACT_MAP: dict[str, tuple[str, str]] = {
    "microbiome_otu": ("microbiome", "sample"),
    "microbiome_deseq": ("microbiome", "sample"),
    "syn_features": ("proteomics", "sample"),
    "syn_total": ("proteomics", "sample"),
    "syn_phospho": ("proteomics", "sample"),
    "syn_ratio": ("proteomics", "sample"),
    "syn_agg": ("proteomics", "sample"),
    "syn_assay": ("clinical_assay", "sample"),
    "syn_labels": ("clinical", "sample"),
    "shap_importance": ("explainability", "feature"),
    "mofa_factors": ("integration_mofa", "subject_id"),
    "snf_clusters": ("integration_snf", "sample_id"),
    "fusion_matrix": ("integration_fusion", "sample"),
    "fusion_cv": ("integration_fusion", "fold"),
    "svc_microbiome": ("ml_predictions", "index"),
    "svc_syn": ("ml_predictions", "index"),
}


@dataclass
class DocumentRecord:
    """One RAG document derived from a pipeline artifact."""

    doc_id: str
    subject_id: str
    work_type: str
    source_file: str
    summary_text: str
    raw_payload: str

    @property
    def collection(self) -> str:
        return WORK_TYPE_TO_COLLECTION.get(self.work_type, self.work_type)


def load_subjects(subjects_file: str | Path) -> list[str]:
    """Load subject IDs from a tab/CSV manifest."""
    path = Path(subjects_file)
    if not path.exists():
        raise FileNotFoundError(f"subjects_file not found: {path}")

    if path.suffix.lower() in {".txt", ".tsv"}:
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path)

    for col in ("sample-id", "sample_id", "sample", "subject_id", "PATNO"):
        if col in df.columns:
            return [str(v) for v in df[col].dropna().unique()]

    first_col = df.columns[0]
    return [str(v) for v in df[first_col].dropna().unique()]


def _resolve_subject_col(df: pd.DataFrame, preferred: str) -> str | None:
    candidates = [
        preferred,
        "sample",
        "sample_id",
        "sample-id",
        "subject_id",
        "PATNO",
        df.columns[0],
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _row_summary(subject_id: str, work_type: str, row: dict[str, Any]) -> str:
    parts = [f"Subject {subject_id} ({work_type})"]
    for key, val in row.items():
        if key in {"sample", "sample_id", "subject_id", "index"}:
            continue
        parts.append(f"{key}={val}")
    return " ".join(parts)


def ingest_csv(
    path: str | Path,
    work_type: str,
    subject_col: str = "sample",
) -> list[DocumentRecord]:
    """Ingest a CSV/TSV file into document records."""
    p = Path(path)
    if p.suffix.lower() == ".tsv":
        df = pd.read_csv(p, sep="\t")
    else:
        df = pd.read_csv(p)

    col = _resolve_subject_col(df, subject_col)
    records: list[DocumentRecord] = []

    if col is None:
        row_dict = df.iloc[0].to_dict() if len(df) else {}
        payload = json.dumps(row_dict, default=str)
        records.append(
            DocumentRecord(
                doc_id=f"{work_type}:COHORT:0",
                subject_id="COHORT",
                work_type=work_type,
                source_file=str(p),
                summary_text=f"Cohort-level {work_type} data from {p.name}",
                raw_payload=payload,
            )
        )
        return records

    for idx, row in df.iterrows():
        subject_id = str(row[col]).strip().strip('"')
        if not subject_id or subject_id.lower() == "nan":
            continue
        row_dict = {k: row[k] for k in df.columns if k != col}
        payload = json.dumps(row_dict, default=str)
        records.append(
            DocumentRecord(
                doc_id=f"{work_type}:{subject_id}:{idx}",
                subject_id=subject_id,
                work_type=work_type,
                source_file=str(p),
                summary_text=_row_summary(subject_id, work_type, row_dict),
                raw_payload=payload,
            )
        )
    return records


def load_prediction_index(index_file: str | Path) -> list[str]:
    """Read subject IDs (column 0) from an SVC test-data file, in row order."""
    p = Path(index_file)
    df = pd.read_csv(p, sep="\t" if p.suffix.lower() == ".tsv" else ",")
    return [str(v).strip().strip('"') for v in df.iloc[:, 0]]


def ingest_svc_predictions(
    path: str | Path,
    work_type: str = "ml_predictions",
    source_key: str = "svc",
    index_file: str | Path | None = None,
) -> list[DocumentRecord]:
    """Ingest bare SVC prediction files (one label per line, no header).

    ``source_key`` distinguishes artifacts that share a ``work_type`` so their
    document IDs stay unique within a single Chroma collection. When
    ``index_file`` is given, its column 0 supplies the real subject IDs; the
    predictions are ordered exactly as the rows of that file. Rows without a
    matching index entry fall back to fabricated ``SAMPLE_nnn`` IDs.
    """
    p = Path(path)
    lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]

    subject_ids: list[str] = []
    if index_file is not None:
        try:
            subject_ids = load_prediction_index(index_file)
        except Exception as exc:
            logger.warning("Failed to read prediction index %s: %s", index_file, exc)
        if subject_ids and len(subject_ids) != len(lines):
            logger.warning(
                "Prediction index %s has %d rows but %s has %d predictions; "
                "unmatched rows keep fabricated IDs",
                index_file,
                len(subject_ids),
                p,
                len(lines),
            )

    records: list[DocumentRecord] = []
    for idx, label in enumerate(lines):
        if idx < len(subject_ids) and subject_ids[idx] and subject_ids[idx].lower() != "nan":
            subject_id = subject_ids[idx]
        else:
            subject_id = f"SAMPLE_{idx + 1:03d}"
        payload = json.dumps(
            {"prediction": label, "line_index": idx, "artifact": source_key}
        )
        records.append(
            DocumentRecord(
                doc_id=f"{work_type}:{source_key}:{subject_id}:{idx}",
                subject_id=subject_id,
                work_type=work_type,
                source_file=str(p),
                summary_text=(
                    f"Subject {subject_id} {work_type} ({source_key}) prediction={label}"
                ),
                raw_payload=payload,
            )
        )
    return records


def ingest_json_dir(path: str | Path) -> list[DocumentRecord]:
    """Ingest Stage1Output-style JSON files from a directory."""
    p = Path(path)
    if not p.is_dir():
        return []

    records: list[DocumentRecord] = []
    for json_path in sorted(p.glob("*.json")):
        data = json.loads(json_path.read_text())
        subject_id = str(data.get("subject_id", json_path.stem))
        payload = json.dumps(data, default=str)
        summary_parts = [
            f"Subject {subject_id} stage1 diagnosis={data.get('diagnosis')}",
            f"confidence={data.get('prediction_confidence')}",
            f"stage={data.get('disease_stage')}",
        ]
        records.append(
            DocumentRecord(
                doc_id=f"stage1:{subject_id}",
                subject_id=subject_id,
                work_type="stage1",
                source_file=str(json_path),
                summary_text=" ".join(summary_parts),
                raw_payload=payload,
            )
        )
    return records


def _prediction_index_for(
    params: dict[str, str], param_key: str, pred_path: Path
) -> Path | None:
    """Locate the test-data file whose row order matches an SVC prediction file.

    Uses the optional ``<param_key>_index`` parameter when present, otherwise
    falls back to the sibling ``test_data.csv`` written by DataSplit.
    """
    explicit = params.get(f"{param_key}_index")
    if explicit:
        candidate = Path(resolve_path(explicit))
        if candidate.exists():
            return candidate
        logger.warning("Missing %s_index file: %s", param_key, candidate)
        return None
    sibling = pred_path.parent / "test_data.csv"
    return sibling if sibling.exists() else None


def ingest_all(params: dict[str, str]) -> list[DocumentRecord]:
    """Ingest all configured pipeline artifacts."""
    records: list[DocumentRecord] = []

    for param_key, (work_type, subject_col) in PARAM_ARTIFACT_MAP.items():
        path_str = params.get(param_key)
        if not path_str:
            continue
        path = Path(resolve_path(path_str))
        if not path.exists():
            logger.warning("Skipping missing artifact %s: %s", param_key, path)
            continue
        try:
            if param_key in {"svc_microbiome", "svc_syn"}:
                records.extend(
                    ingest_svc_predictions(
                        path,
                        "ml_predictions",
                        source_key=param_key,
                        index_file=_prediction_index_for(params, param_key, path),
                    )
                )
            elif param_key == "stage1_json_dir":
                records.extend(ingest_json_dir(path))
            else:
                records.extend(ingest_csv(path, work_type, subject_col))
        except Exception as exc:
            logger.warning("Failed to ingest %s (%s): %s", param_key, path, exc)

    stage1_dir = params.get("stage1_json_dir")
    if stage1_dir and "stage1_json_dir" not in PARAM_ARTIFACT_MAP:
        records.extend(ingest_json_dir(resolve_path(stage1_dir)))

    return records


def records_to_dicts(records: list[DocumentRecord]) -> list[dict[str, str]]:
    """Serialize records for tests/debugging."""
    return [asdict(r) for r in records]
