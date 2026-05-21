"""
Model catalog: Hugging Face provenance mapped to Ollama tags.

Author: Joseph R. Quinn <quinn.josephr@protonmail.com>
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass

from hardware import HardwareInfo


@dataclass(frozen=True)
class ModelTier:
    """One selectable Ollama model tier."""

    ollama_tag: str
    hf_repo: str
    min_vram_gb: float
    min_ram_gb: float
    description: str


MODEL_TIERS: list[ModelTier] = [
    ModelTier(
        ollama_tag="gemma4:31b",
        hf_repo="google/gemma-4-31B",
        min_vram_gb=24.0,
        min_ram_gb=32.0,
        description="Gemma 4 31B dense — highest quality",
    ),
    ModelTier(
        ollama_tag="gemma4:26b",
        hf_repo="google/gemma-4-26B-A4B",
        min_vram_gb=16.0,
        min_ram_gb=24.0,
        description="Gemma 4 26B MoE — strong quality, efficient active params",
    ),
    ModelTier(
        ollama_tag="gemma4:e4b",
        hf_repo="google/gemma-4-E4B-IT",
        min_vram_gb=10.0,
        min_ram_gb=16.0,
        description="Gemma 4 E4B instruction-tuned — default GPU tier",
    ),
    ModelTier(
        ollama_tag="gemma4:e2b",
        hf_repo="google/gemma-4-E2B-IT",
        min_vram_gb=8.0,
        min_ram_gb=12.0,
        description="Gemma 4 E2B instruction-tuned — low VRAM",
    ),
    ModelTier(
        ollama_tag="llama3.1:8b",
        hf_repo="meta-llama/Meta-Llama-3.1-8B-Instruct",
        min_vram_gb=8.0,
        min_ram_gb=12.0,
        description="Llama 3.1 8B fallback when Gemma 4 unavailable",
    ),
]

FALLBACK_MODEL = "llama3.1:8b"


def select_model(hw: HardwareInfo) -> str:
    """Select the best Ollama model tag for detected hardware."""
    available_gb = hw.gpu_vram_gb if hw.gpu_available and hw.gpu_vram_gb else hw.ram_gb * 0.75

    for tier in MODEL_TIERS:
        mem_required = tier.min_vram_gb if hw.gpu_available else tier.min_ram_gb
        if available_gb >= mem_required:
            return tier.ollama_tag

    if hw.gpu_available:
        for tier in reversed(MODEL_TIERS):
            if tier.ollama_tag.startswith("gemma4"):
                return tier.ollama_tag

    return FALLBACK_MODEL


def get_hf_repo(ollama_tag: str) -> str | None:
    """Return Hugging Face repo ID for an Ollama tag, if known."""
    for tier in MODEL_TIERS:
        if tier.ollama_tag == ollama_tag:
            return tier.hf_repo
    return None
