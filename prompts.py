"""
Prompt templates for clinical and scientific analysis.

Author: Joseph R. Quinn <quinn.josephr@protonmail.com>
License: MIT
"""

from __future__ import annotations

from typing import Any


def build_scientific_prompt(subject_id: str, rag_context: str) -> str:
    """Build scientific analysis prompt."""
    return f"""You are a multi-omics bioinformatics expert analyzing Parkinson's disease pipeline results.

Subject ID: {subject_id}

{rag_context}

Provide a rigorous scientific analysis covering:
1. Key biomarkers and their cross-modal patterns
2. Integration method findings (MOFA, SNF, fusion if present)
3. Statistical/explainability signals (SHAP, classifier outputs)
4. Data limitations and confounders

Write 2-4 paragraphs in plain scientific prose. Do not invent data not present in the context."""


def build_clinical_prompt(subject_id: str, rag_context: str, scientific_analysis: str) -> str:
    """Build clinical interpretation prompt."""
    return f"""You are a movement disorders clinician interpreting research pipeline outputs for subject {subject_id}.

Scientific analysis already performed:
{scientific_analysis}

Additional pipeline context:
{rag_context}

Provide a clinical interpretation covering:
1. What the findings suggest about Parkinson's pathology for this subject
2. Actionable considerations for a research/clinical team (not treatment prescriptions)
3. Uncertainty and what additional data would help

Write 2-3 paragraphs accessible to a clinician. Do not invent measurements."""


def build_pd_likelihood_prompt(
    subject_id: str,
    rag_context: str,
    structured_signals: dict[str, Any],
) -> str:
    """Build PD likelihood assessment prompt."""
    signals_text = "\n".join(f"- {k}: {v}" for k, v in structured_signals.items())
    return f"""You are assessing Parkinson's disease likelihood for research subject {subject_id}.

Structured pre-scores and signals:
{signals_text}

Pipeline RAG context:
{rag_context}

Respond with ONLY valid JSON (no markdown fences) using this schema:
{{
  "subject_id": "{subject_id}",
  "label": "likely_PD" | "unlikely_PD" | "indeterminate",
  "confidence": 0.0 to 1.0,
  "supporting_evidence": ["..."],
  "contradicting_evidence": ["..."],
  "model_basis": "brief description of evidence used",
  "narrative": "2-3 sentence plain-language summary for clinicians"
}}

Base the label on available pipeline evidence only. Use indeterminate when evidence is mixed or sparse."""


def build_modality_prompt(subject_id: str, work_type: str, rag_context: str) -> str:
    """Build per-modality findings summary."""
    return f"""Summarize {work_type} findings for Parkinson's research subject {subject_id}.

{rag_context}

Write 1-2 concise paragraphs. Only describe evidence present in the context."""
