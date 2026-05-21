import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from artifact_ingest import ingest_all, ingest_csv, ingest_json_dir, load_subjects

SYNTHETIC = Path(__file__).parent.parent / "example" / "synthetic"


def test_load_subjects():
    subjects = load_subjects(SYNTHETIC / "Samples.Syn.txt")
    assert "PD_001" in subjects
    assert "CTRL_001" in subjects


def test_ingest_features_csv():
    records = ingest_csv(SYNTHETIC / "features.csv", "proteomics", "sample")
    assert len(records) == 3
    assert all(r.work_type == "proteomics" for r in records)
    pd_rec = next(r for r in records if r.subject_id == "PD_001")
    assert "ps129_ratio" in pd_rec.summary_text


def test_ingest_stage1_json():
    records = ingest_json_dir(SYNTHETIC / "stage1")
    assert len(records) == 2
    assert all(r.work_type == "stage1" for r in records)
    pd_rec = next(r for r in records if r.subject_id == "PD_001")
    assert "PD" in pd_rec.summary_text


def test_ingest_all_skips_missing(caplog):
    params = {
        "subjects_file": str(SYNTHETIC / "Samples.Syn.txt"),
        "syn_features": str(SYNTHETIC / "features.csv"),
        "shap_importance": str(SYNTHETIC / "nonexistent.csv"),
        "stage1_json_dir": str(SYNTHETIC / "stage1"),
    }
    with caplog.at_level(logging.WARNING):
        records = ingest_all(params)
    assert any(r.work_type == "proteomics" for r in records)
    assert any(r.work_type == "stage1" for r in records)
    assert "Skipping missing artifact" in caplog.text
