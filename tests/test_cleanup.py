import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from cleanup import cleanup_session
from ollama_lifecycle import OllamaSession


def test_cleanup_removes_chroma(tmp_path):
    session = OllamaSession(work_dir=tmp_path)
    chroma = tmp_path / "chroma"
    chroma.mkdir()
    (chroma / "data").write_text("x")
    cleanup_session(session, {"keep_chroma": "false", "ollama_auto_stop": "false"})
    assert not chroma.exists()


def test_cleanup_keeps_chroma(tmp_path):
    session = OllamaSession(work_dir=tmp_path)
    chroma = tmp_path / "chroma"
    chroma.mkdir()
    cleanup_session(session, {"keep_chroma": "true", "ollama_auto_stop": "false"})
    assert chroma.exists()


@patch("cleanup.stop_ollama_if_started")
def test_cleanup_calls_ollama_stop(mock_stop, tmp_path):
    session = OllamaSession(work_dir=tmp_path, started_by_plugin=True, server_pid=99)
    cleanup_session(session, {"keep_chroma": "true", "ollama_auto_stop": "true"})
    mock_stop.assert_called_once()
