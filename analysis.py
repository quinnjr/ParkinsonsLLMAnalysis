"""
Per-subject RAG + LLM analysis orchestration.

Author: Joseph R. Quinn <quinn.josephr@protonmail.com>
License: MIT
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from chroma_store import format_rag_context, query_subject
from prompts import (
    build_clinical_prompt,
    build_modality_prompt,
    build_pd_likelihood_prompt,
    build_scientific_prompt,
)

SCIENTIFIC_WORK_TYPES = [
    "microbiome",
    "proteomics",
    "integration_mofa",
    "integration_snf",
    "integration_fusion",
    "explainability",
    "stage1",
]

CLINICAL_WORK_TYPES = [
    "clinical_assay",
    "clinical",
    "ml_predictions",
    "proteomics",
    "stage1",
]

MODALITY_SECTIONS = [
    ("microbiome", "Microbiome Findings"),
    ("proteomics", "Proteomics and Alpha-Synuclein Findings"),
    ("integration_mofa", "MOFA+ Integration Results"),
    ("integration_snf", "SNF Integration Results"),
    ("integration_fusion", "Early Fusion Results"),
    ("explainability", "SHAP Explainability Highlights"),
]


@dataclass
class PDLikelihood:
    """Structured PD likelihood assessment."""

    subject_id: str
    label: Literal["likely_PD", "unlikely_PD", "indeterminate"]
    confidence: float
    supporting_evidence: list[str]
    contradicting_evidence: list[str]
    model_basis: str
    narrative: str = ""


def _extract_json(text: str) -> dict[str, Any]:
    """Extract JSON object from LLM response."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


def _precompute_signals(subject_id: str, db_path: Path) -> dict[str, Any]:
    """Deterministic pre-scores from ingested pipeline data."""
    chunks = query_subject(
        db_path,
        subject_id,
        ["proteomics", "clinical_assay", "ml_predictions", "stage1"],
        f"{subject_id} Parkinson prediction biomarkers",
        k=10,
    )
    signals: dict[str, Any] = {"subject_id": subject_id}

    for chunk in chunks:
        meta = chunk.get("metadata", {})
        payload_raw = meta.get("raw_payload", "")
        try:
            payload = json.loads(payload_raw) if payload_raw else {}
        except json.JSONDecodeError:
            payload = {}

        wt = chunk.get("work_type")
        if wt == "stage1":
            signals["stage1_diagnosis"] = payload.get("diagnosis")
            signals["stage1_confidence"] = payload.get("prediction_confidence")
            signals["stage1_stage"] = payload.get("disease_stage")
        if wt == "ml_predictions":
            signals.setdefault("svc_predictions", []).append(payload.get("prediction"))
        if wt == "proteomics":
            for key in ("ps129_ratio", "alpha_synuclein_total", "agg_index", "concentration"):
                if key in payload:
                    signals[key] = payload[key]

    pd_votes = 0
    hc_votes = 0
    if str(signals.get("stage1_diagnosis", "")).upper() in {"PD", "PARKINSONS"}:
        pd_votes += 2
    if str(signals.get("stage1_diagnosis", "")).upper() in {"HC", "CONTROL", "HEALTHY"}:
        hc_votes += 2
    for pred in signals.get("svc_predictions", []):
        p = str(pred).lower()
        if "park" in p or p in {"1", "pd"}:
            pd_votes += 1
        elif "control" in p or "hc" in p or p in {"0"}:
            hc_votes += 1
    if subject_id.upper().startswith("PD"):
        pd_votes += 1
    if subject_id.upper().startswith(("CTRL", "HC")):
        hc_votes += 1

    signals["pd_vote_score"] = pd_votes
    signals["hc_vote_score"] = hc_votes
    return signals


def parse_pd_likelihood(text: str, subject_id: str) -> PDLikelihood:
    """Parse LLM JSON into PDLikelihood."""
    data = _extract_json(text)
    label = data.get("label", "indeterminate")
    if label not in {"likely_PD", "unlikely_PD", "indeterminate"}:
        label = "indeterminate"
    return PDLikelihood(
        subject_id=subject_id,
        label=label,
        confidence=float(data.get("confidence", 0.5)),
        supporting_evidence=list(data.get("supporting_evidence", [])),
        contradicting_evidence=list(data.get("contradicting_evidence", [])),
        model_basis=str(data.get("model_basis", "pipeline RAG + LLM synthesis")),
        narrative=str(data.get("narrative", "")),
    )


def analyze_subject(
    subject_id: str,
    db_path: str | Path,
    model: str,
    params: dict[str, str],
    generate_fn: Callable[[str], str],
) -> dict[str, Any]:
    """Run full per-subject analysis pipeline."""
    db_path = Path(db_path)
    structured_signals = _precompute_signals(subject_id, db_path)

    sci_chunks = query_subject(
        db_path,
        subject_id,
        SCIENTIFIC_WORK_TYPES,
        f"{subject_id} multi-omics Parkinson biomarkers integration",
        k=8,
    )
    sci_context = format_rag_context(sci_chunks)
    scientific_analysis = generate_fn(build_scientific_prompt(subject_id, sci_context))

    clin_chunks = query_subject(
        db_path,
        subject_id,
        CLINICAL_WORK_TYPES,
        f"{subject_id} clinical assay Parkinson diagnosis",
        k=8,
    )
    clin_context = format_rag_context(clin_chunks)
    clinical_analysis = generate_fn(
        build_clinical_prompt(subject_id, clin_context, scientific_analysis)
    )

    pd_prompt = build_pd_likelihood_prompt(subject_id, sci_context + "\n" + clin_context, structured_signals)
    pd_raw = generate_fn(pd_prompt)
    pd_likelihood = parse_pd_likelihood(pd_raw, subject_id)

    modality_sections: dict[str, str] = {}
    for work_type, _title in MODALITY_SECTIONS:
        mod_chunks = query_subject(
            db_path,
            subject_id,
            [work_type],
            f"{subject_id} {work_type} Parkinson findings",
            k=5,
        )
        if mod_chunks:
            modality_sections[work_type] = generate_fn(
                build_modality_prompt(subject_id, work_type, format_rag_context(mod_chunks))
            )
        else:
            modality_sections[work_type] = "No data available for this modality."

    return {
        "subject_id": subject_id,
        "model": model,
        "structured_signals": structured_signals,
        "scientific_analysis": scientific_analysis,
        "clinical_analysis": clinical_analysis,
        "pd_likelihood": asdict(pd_likelihood),
        "modality_sections": modality_sections,
        "rag_chunk_count": len(sci_chunks) + len(clin_chunks),
    }
