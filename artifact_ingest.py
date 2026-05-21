"""
Ingest Parkinson's pipeline artifacts into DocumentRecord objects.

Author: Joseph R. Quinn <quinn.josephr@protonmail.com>
License: MIT
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

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


def ingest_svc_predictions(path: str | Path, work_type: str = "ml_predictions") -> list[DocumentRecord]:
    """Ingest bare SVC prediction files (one label per line, no header)."""
    p = Path(path)
    lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    records: list[DocumentRecord] = []
    for idx, label in enumerate(lines):
        subject_id = f"SAMPLE_{idx + 1:03d}"
        payload = json.dumps({"prediction": label, "line_index": idx})
        records.append(
            DocumentRecord(
                doc_id=f"{work_type}:{subject_id}:{idx}",
                subject_id=subject_id,
                work_type=work_type,
                source_file=str(p),
                summary_text=f"Subject {subject_id} ml_predictions prediction={label}",
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


def ingest_all(params: dict[str, str]) -> list[DocumentRecord]:
    """Ingest all configured pipeline artifacts."""
    records: list[DocumentRecord] = []

    for param_key, (work_type, subject_col) in PARAM_ARTIFACT_MAP.items():
        path_str = params.get(param_key)
        if not path_str:
            continue
        path = Path(path_str)
        if not path.exists():
            logger.warning("Skipping missing artifact %s: %s", param_key, path)
            continue
        try:
            if param_key in {"svc_microbiome", "svc_syn"}:
                records.extend(ingest_svc_predictions(path, "ml_predictions"))
            elif param_key == "stage1_json_dir":
                records.extend(ingest_json_dir(path))
            else:
                records.extend(ingest_csv(path, work_type, subject_col))
        except Exception as exc:
            logger.warning("Failed to ingest %s (%s): %s", param_key, path, exc)

    stage1_dir = params.get("stage1_json_dir")
    if stage1_dir and "stage1_json_dir" not in PARAM_ARTIFACT_MAP:
        records.extend(ingest_json_dir(stage1_dir))

    return records


def records_to_dicts(records: list[DocumentRecord]) -> list[dict[str, str]]:
    """Serialize records for tests/debugging."""
    return [asdict(r) for r in records]
