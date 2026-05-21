"""
Session cleanup: temp artifacts and Ollama shutdown.

Author: Joseph R. Quinn <quinn.josephr@protonmail.com>
License: MIT
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ollama_lifecycle import OllamaSession, stop_ollama_if_started


def _param_bool(params: dict[str, str], key: str, default: bool = False) -> bool:
    return params.get(key, str(default).lower()).lower() == "true"


def cleanup_session(session: OllamaSession, params: dict[str, str]) -> None:
    """Remove temp artifacts and stop Ollama if we started it."""
    keep_chroma = _param_bool(params, "keep_chroma", False)
    keep_temp_html = _param_bool(params, "keep_temp_html", False)

    chroma_dir = session.work_dir / "chroma"
    if chroma_dir.exists() and not keep_chroma:
        shutil.rmtree(chroma_dir, ignore_errors=True)

    if not keep_temp_html:
        for temp in session.temp_paths:
            if temp.exists():
                if temp.is_dir():
                    shutil.rmtree(temp, ignore_errors=True)
                else:
                    temp.unlink(missing_ok=True)
        for html_file in session.work_dir.glob("**/*.html"):
            if html_file.name.endswith(".html"):
                html_file.unlink(missing_ok=True)

    stop_ollama_if_started(session, params)

    if session.work_dir.exists() and not any(session.work_dir.iterdir()):
        session.work_dir.rmdir()
