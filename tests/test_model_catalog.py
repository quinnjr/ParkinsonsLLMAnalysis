import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from hardware import HardwareInfo
from model_catalog import FALLBACK_MODEL, select_model
from ollama_lifecycle import OllamaSession, stop_ollama_if_started


def test_select_model_high_vram():
    hw = HardwareInfo(16, 64.0, True, "GPU", 24.0)
    assert select_model(hw) == "gemma4:31b"


def test_select_model_low_vram():
    hw = HardwareInfo(8, 16.0, True, "GPU", 6.0)
    assert select_model(hw) == "gemma4:e2b"


def test_select_model_cpu_only():
    hw = HardwareInfo(4, 8.0, False, None, None)
    assert select_model(hw) == FALLBACK_MODEL


@patch("ollama_lifecycle.is_model_available", return_value=False)
@patch("ollama_lifecycle.download_model")
def test_ensure_model_pulls(mock_download, mock_available):
    from ollama_lifecycle import ensure_model

    ensure_model("gemma4:e4b")
    mock_download.assert_called_once_with("gemma4:e4b")


@patch("os.kill")
def test_stop_ollama_only_when_started(mock_kill):
    session = OllamaSession(work_dir=Path("/tmp/x"), started_by_plugin=True, server_pid=1234)
    stop_ollama_if_started(session, {"ollama_auto_stop": "true"})
    mock_kill.assert_called()


@patch("os.kill")
def test_stop_ollama_skips_external(mock_kill):
    session = OllamaSession(work_dir=Path("/tmp/x"), started_by_plugin=False, server_pid=1234)
    stop_ollama_if_started(session, {"ollama_auto_stop": "true"})
    mock_kill.assert_not_called()
