import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from artifact_ingest import (
    ingest_all,
    ingest_csv,
    ingest_json_dir,
    ingest_svc_predictions,
    load_subjects,
)

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


def test_svc_predictions_use_index_and_unique_ids(tmp_path):
    (tmp_path / "CSV").mkdir()
    (tmp_path / "Syn").mkdir()
    for sub in ("CSV", "Syn"):
        (tmp_path / sub / "output_svc.csv").write_text("PD\nControl\n")
        (tmp_path / sub / "test_data.csv").write_text(
            "sample,f1\nPD_001,1.0\nCTRL_001,2.0\n"
        )

    params = {
        "svc_microbiome": str(tmp_path / "CSV" / "output_svc.csv"),
        "svc_syn": str(tmp_path / "Syn" / "output_svc.csv"),
    }
    records = ingest_all(params)

    doc_ids = [r.doc_id for r in records]
    assert len(doc_ids) == len(set(doc_ids))
    assert {r.subject_id for r in records} == {"PD_001", "CTRL_001"}
    assert all(r.work_type == "ml_predictions" for r in records)


def test_svc_predictions_fall_back_to_fabricated_ids(tmp_path):
    pred = tmp_path / "output_svc.csv"
    pred.write_text("PD\nControl\n")
    records = ingest_svc_predictions(pred, source_key="svc_syn")
    assert [r.subject_id for r in records] == ["SAMPLE_001", "SAMPLE_002"]
    assert records[0].doc_id.startswith("ml_predictions:svc_syn:")
