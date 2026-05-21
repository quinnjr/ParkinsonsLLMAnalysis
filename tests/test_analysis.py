import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis import analyze_subject, parse_pd_likelihood
from prompts import build_pd_likelihood_prompt

SYNTHETIC = Path(__file__).parent.parent / "example" / "synthetic"


def test_build_pd_likelihood_prompt_contains_subject():
    prompt = build_pd_likelihood_prompt(
        "PD_001", "context here", {"pd_vote_score": 2, "stage1_diagnosis": "PD"},
    )
    assert "PD_001" in prompt
    assert "pd_vote_score" in prompt


def test_parse_pd_likelihood():
    raw = json.dumps({
        "subject_id": "PD_001",
        "label": "likely_PD",
        "confidence": 0.85,
        "supporting_evidence": ["elevated ps129_ratio"],
        "contradicting_evidence": [],
        "model_basis": "test",
        "narrative": "Likely PD based on biomarkers.",
    })
    result = parse_pd_likelihood(raw, "PD_001")
    assert result.label == "likely_PD"
    assert result.confidence == 0.85


def test_analyze_subject_mocked(tmp_path):
    from artifact_ingest import ingest_all
    from chroma_store import build_store

    params = {
        "syn_features": str(SYNTHETIC / "features.csv"),
        "stage1_json_dir": str(SYNTHETIC / "stage1"),
    }
    docs = ingest_all(params)
    db_path = tmp_path / "chroma"
    build_store(docs, db_path)

    pd_json = json.dumps({
        "subject_id": "PD_001",
        "label": "likely_PD",
        "confidence": 0.9,
        "supporting_evidence": ["a"],
        "contradicting_evidence": [],
        "model_basis": "mock",
        "narrative": "Test narrative.",
    })

    call_count = 0

    def mock_generate(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        if "ONLY valid JSON" in prompt:
            return pd_json
        return f"Analysis paragraph {call_count} for PD_001."

    result = analyze_subject("PD_001", db_path, "gemma4:e4b", {}, mock_generate)
    assert "scientific_analysis" in result
    assert "clinical_analysis" in result
    assert result["pd_likelihood"]["label"] == "likely_PD"
    assert call_count >= 3
