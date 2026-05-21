"""
Hardware detection for model selection.

Author: Joseph R. Quinn <quinn.josephr@protonmail.com>
License: MIT
"""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass


@dataclass
class HardwareInfo:
    """System hardware snapshot for LLM model selection."""

    cpu_cores: int
    ram_gb: float
    gpu_available: bool
    gpu_name: str | None
    gpu_vram_gb: float | None

    def __str__(self) -> str:
        gpu = (
            f"{self.gpu_name} ({self.gpu_vram_gb:.1f}GB VRAM)"
            if self.gpu_available and self.gpu_vram_gb
            else "None"
        )
        return f"CPU: {self.cpu_cores} cores | RAM: {self.ram_gb:.1f}GB | GPU: {gpu}"


def _read_ram_gb() -> float:
    try:
        import psutil

        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024**2)
    except OSError:
        pass
    return 8.0


def _detect_nvidia() -> tuple[bool, str | None, float | None]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return False, None, None
        line = result.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        name = parts[0]
        vram = float(parts[1]) / 1024 if len(parts) > 1 else None
        return True, name, vram
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return False, None, None


def _detect_rocm() -> tuple[bool, str | None, float | None]:
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0:
            return False, None, None
        for line in result.stdout.splitlines():
            if "Total Memory" in line or "VRAM Total" in line:
                digits = "".join(c if c.isdigit() or c == "." else " " for c in line)
                values = [float(x) for x in digits.split() if x]
                if values:
                    return True, "AMD GPU", values[0] / 1024
        return True, "AMD GPU", 8.0
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return False, None, None


def _detect_apple_gpu() -> tuple[bool, str | None, float | None]:
    if platform.system() != "Darwin":
        return False, None, None
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.optional.arm64"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.stdout.strip() == "1":
            return True, "Apple Silicon", 16.0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False, None, None


def has_gpu(info: HardwareInfo | None = None) -> bool:
    """Return True if a GPU is available."""
    if info is None:
        info = detect_hardware()
    return info.gpu_available


def detect_hardware() -> HardwareInfo:
    """Detect CPU, RAM, and GPU capabilities."""
    cpu_cores = os.cpu_count() or 1
    ram_gb = _read_ram_gb()

    gpu_available, gpu_name, gpu_vram = _detect_nvidia()
    if not gpu_available:
        gpu_available, gpu_name, gpu_vram = _detect_rocm()
    if not gpu_available:
        gpu_available, gpu_name, gpu_vram = _detect_apple_gpu()

    if gpu_available and gpu_vram is None:
        gpu_vram = max(ram_gb * 0.5, 8.0)

    return HardwareInfo(
        cpu_cores=cpu_cores,
        ram_gb=ram_gb,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram,
    )
