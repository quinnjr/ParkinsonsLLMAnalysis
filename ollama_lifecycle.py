"""
Ollama server lifecycle and model management.

Author: Joseph R. Quinn <quinn.josephr@protonmail.com>
License: MIT
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class OllamaSession:
    """Tracks Ollama process ownership and temp artifacts."""

    work_dir: Path
    started_by_plugin: bool = False
    server_pid: int | None = None
    temp_paths: list[Path] = field(default_factory=list)


def is_ollama_running() -> bool:
    """Return True if Ollama API responds."""
    try:
        import ollama

        ollama.list()
        return True
    except Exception:
        return False


def _ollama_version() -> tuple[int, int, int] | None:
    """Parse ollama --version if available."""
    ollama_path = shutil.which("ollama")
    if not ollama_path:
        return None
    try:
        result = subprocess.run(
            [ollama_path, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        text = (result.stdout or result.stderr).strip()
        for token in text.split():
            if token[0].isdigit() and "." in token:
                parts = token.split(".")
                nums = [int(p) for p in parts[:3] if p.isdigit()]
                while len(nums) < 3:
                    nums.append(0)
                return nums[0], nums[1], nums[2]
    except (subprocess.TimeoutExpired, ValueError):
        pass
    return None


def _supports_gemma4() -> bool:
    version = _ollama_version()
    if version is None:
        return True
    major, minor, _ = version
    return major > 0 or minor >= 20


def start_ollama_server(use_gpu: bool = True, timeout: int = 30) -> subprocess.Popen[Any]:
    """Start Ollama serve as a background process."""
    ollama_path = shutil.which("ollama")
    if ollama_path is None:
        raise RuntimeError(
            "Ollama executable not found. Install from https://ollama.com/download"
        )

    env = os.environ.copy()
    if use_gpu:
        env.setdefault("OLLAMA_NUM_GPU", "999")

    proc = subprocess.Popen(
        [ollama_path, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )

    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_ollama_running():
            return proc
        if proc.poll() is not None:
            raise RuntimeError("Ollama server exited during startup")
        time.sleep(0.5)

    proc.terminate()
    raise RuntimeError(f"Ollama server did not start within {timeout}s")


def ensure_ollama_running(session: OllamaSession, params: dict[str, str]) -> OllamaSession:
    """Ensure Ollama is running; start it if auto-start is enabled."""
    auto_start = params.get("ollama_auto_start", "true").lower() == "true"
    if is_ollama_running():
        return session
    if not auto_start:
        raise RuntimeError("Ollama is not running and ollama_auto_start is false")
    proc = start_ollama_server(use_gpu=True)
    session.started_by_plugin = True
    session.server_pid = proc.pid
    return session


def list_local_models() -> dict[str, dict]:
    """Return locally available Ollama models keyed by name."""
    import ollama

    response = ollama.list()
    models: dict[str, dict] = {}
    for model in response.get("models", []):
        name = model.get("name", "")
        models[name] = model
        base = name.split(":")[0]
        if base not in models:
            models[base] = model
    return models


def is_model_available(model_name: str, local_models: dict[str, dict] | None = None) -> bool:
    """Return True if model is available locally."""
    if local_models is None:
        local_models = list_local_models()
    if model_name in local_models:
        return True
    if f"{model_name}:latest" in local_models:
        return True
    base = model_name.split(":")[0]
    return base in local_models


def download_model(model_name: str) -> None:
    """Pull model via Ollama."""
    import ollama

    if model_name.startswith("gemma4") and not _supports_gemma4():
        raise RuntimeError(
            "Gemma 4 requires Ollama >= 0.20. Upgrade Ollama or set model_name llama3.1:8b"
        )

    for progress in ollama.pull(model_name, stream=True):
        status = progress.get("status", "")
        completed = progress.get("completed", 0)
        total = progress.get("total", 0)
        if total > 0:
            pct = (completed / total) * 100
            print(f"\r  {status}: {pct:.1f}%", end="", flush=True)
        else:
            print(f"\r  {status}...", end="", flush=True)
    print()


def ensure_model(model_name: str) -> None:
    """Download model if not present locally."""
    if is_model_available(model_name):
        return
    download_model(model_name)


def generate(
    model: str,
    prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> str:
    """Generate completion text via Ollama."""
    import ollama

    response = ollama.generate(
        model=model,
        prompt=prompt,
        options={"temperature": temperature, "num_predict": max_tokens},
    )
    return response.get("response", "")


def stop_ollama_if_started(session: OllamaSession, params: dict[str, str]) -> None:
    """Stop Ollama only if this plugin started it."""
    auto_stop = params.get("ollama_auto_stop", "true").lower() == "true"
    if not auto_stop or not session.started_by_plugin:
        return
    if session.server_pid is None:
        return
    try:
        os.kill(session.server_pid, signal.SIGTERM)
        time.sleep(1)
        try:
            os.kill(session.server_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    except ProcessLookupError:
        pass
