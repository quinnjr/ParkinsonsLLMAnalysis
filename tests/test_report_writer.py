import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from report_writer import render_cohort_report, render_subject_report


def test_render_subject_report_sections():
    result = {
        "subject_id": "PD_001",
        "rag_chunk_count": 5,
        "scientific_analysis": "Scientific text.",
        "clinical_analysis": "Clinical text.",
        "pd_likelihood": {
            "label": "likely_PD",
            "confidence": 0.9,
            "model_basis": "test",
            "supporting_evidence": ["ev1"],
            "contradicting_evidence": [],
            "narrative": "Likely PD.",
        },
        "modality_sections": {
            "microbiome": "Microbiome text.",
            "proteomics": "Proteomics text.",
        },
    }
    md = render_subject_report("PD_001", result, {"model": "test", "hardware": "cpu", "timestamp": "t"})
    assert "## Parkinson's Likelihood Assessment" in md
    assert "## Microbiome Findings" in md
    assert "## Proteomics and Alpha-Synuclein Findings" in md
    assert "## Scientific Analysis" in md
    assert "## Clinical Analysis" in md


def test_render_cohort_report():
    cohort = {"subject_count": 1, "subjects": ["PD_001"]}
    results = {"PD_001": {"pd_likelihood": {"label": "likely_PD", "confidence": 0.9, "narrative": "x"}}}
    md = render_cohort_report(cohort, results, {"model": "m", "hardware": "h", "timestamp": "t"})
    assert "PD_001" in md
    assert "likely_PD" in md
