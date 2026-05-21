import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ParkinsonsLLMAnalysis import ParkinsonsLLMAnalysis

SYNTHETIC = Path(__file__).parent.parent / "example" / "synthetic"


def test_plugin_has_pluma_lifecycle():
    p = ParkinsonsLLMAnalysis()
    assert callable(getattr(p, "input", None))
    assert callable(getattr(p, "run", None))
    assert callable(getattr(p, "output", None))


@patch("ParkinsonsLLMAnalysis.ensure_model")
@patch("ParkinsonsLLMAnalysis.ensure_ollama_running")
@patch("ParkinsonsLLMAnalysis.cleanup_session")
def test_end_to_end_mocked(mock_cleanup, mock_ensure_ollama, mock_ensure_model, tmp_path):
    param_file = tmp_path / "params.txt"
    work_dir = tmp_path / "work"
    param_file.write_text(
        f"subjects_file {SYNTHETIC / 'Samples.Syn.txt'}\n"
        f"work_dir {work_dir}\n"
        f"syn_features {SYNTHETIC / 'features.csv'}\n"
        f"stage1_json_dir {SYNTHETIC / 'stage1'}\n"
        "model_name gemma4:e4b\n"
        "ollama_auto_start false\n"
        "ollama_auto_stop false\n"
        "output_pdf false\n"
        "keep_chroma false\n",
        encoding="utf-8",
    )

    pd_json = json.dumps({
        "subject_id": "PD_001",
        "label": "likely_PD",
        "confidence": 0.9,
        "supporting_evidence": ["elevated ps129_ratio"],
        "contradicting_evidence": [],
        "model_basis": "mock",
        "narrative": "Likely PD.",
    })

    def mock_generate(prompt: str) -> str:
        if "ONLY valid JSON" in prompt:
            return pd_json
        return "Mock analysis paragraph."

    plugin = ParkinsonsLLMAnalysis()
    plugin.input(str(param_file))

    with patch.object(plugin, "_generate", side_effect=mock_generate):
        plugin.run()

    out_base = tmp_path / "report"
    plugin.output(str(out_base))

    assert (tmp_path / "report_PD_001.md").exists()
    assert (tmp_path / "report_PD_001.json").exists()
    assert (tmp_path / "report_cohort.md").exists()
    mock_cleanup.assert_called_once()
