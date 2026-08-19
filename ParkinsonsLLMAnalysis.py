"""
ParkinsonsLLMAnalysis PluMA Plugin.

Hardware-aware local LLM analysis of Parkinson's pipeline outputs using Ollama,
ChromaDB RAG over ingested artifacts, and Markdown/PDF report generation.

Author: Joseph R. Quinn <quinn.josephr@protonmail.com>
License: MIT
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from analysis import analyze_subject
from artifact_ingest import ingest_all, load_subjects, resolve_path
from chroma_store import build_store
from cleanup import cleanup_session
from hardware import detect_hardware
from model_catalog import select_model
from ollama_lifecycle import OllamaSession, ensure_model, ensure_ollama_running
from pdf_writer import write_pdf
from report_writer import render_cohort_report, render_subject_report

logger = logging.getLogger(__name__)


def _parse_params(filename: str) -> dict[str, str]:
    """Parse whitespace-delimited key-value parameter file."""
    params: dict[str, str] = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                params[parts[0]] = parts[1]
    return params


def _param_bool(params: dict[str, str], key: str, default: bool = False) -> bool:
    return params.get(key, str(default).lower()).lower() == "true"


class ParkinsonsLLMAnalysis:
    """PluMA plugin for RAG-augmented LLM analysis of Parkinson's pipeline results."""

    def __init__(self) -> None:
        self.parameters: dict[str, str] = {}
        self._results: dict[str, dict[str, Any]] = {}
        self._cohort_result: dict[str, Any] = {}
        self._model_name: str = ""
        self._hardware_summary: str = ""
        self._run_timestamp: str = ""
        self._output_base: str = ""

    def input(self, filename: str) -> None:
        """Load parameter file."""
        self.parameters = _parse_params(filename)

    def run(self) -> None:
        """Execute hardware detection, ingestion, RAG, and per-subject analysis."""
        work_dir = Path(resolve_path(self.parameters["work_dir"]))
        work_dir.mkdir(parents=True, exist_ok=True)
        session = OllamaSession(work_dir=work_dir)
        self._run_timestamp = datetime.now(UTC).isoformat()

        try:
            hw = detect_hardware()
            self._hardware_summary = str(hw)

            model = self.parameters.get("model_name", "auto")
            if model == "auto":
                model = select_model(hw)
            self._model_name = model

            session = ensure_ollama_running(session, self.parameters)
            ensure_model(model)

            subjects = load_subjects(resolve_path(self.parameters["subjects_file"]))
            docs = ingest_all(self.parameters)
            db_path = session.work_dir / "chroma"
            build_store(docs, db_path)

            for subject_id in subjects:
                result = analyze_subject(
                    subject_id=subject_id,
                    db_path=db_path,
                    model=model,
                    params=self.parameters,
                    generate_fn=self._generate,
                )
                self._results[subject_id] = result

            self._cohort_result = {
                "subject_count": len(subjects),
                "subjects": list(subjects),
                "model": model,
                "hardware": self._hardware_summary,
                "timestamp": self._run_timestamp,
                "pd_likelihood_summary": {
                    sid: self._results[sid].get("pd_likelihood", {})
                    for sid in subjects
                },
            }
        finally:
            cleanup_session(session, self.parameters)

    def _generate(self, prompt: str) -> str:
        """Generate text via Ollama (used by analysis module)."""
        from ollama_lifecycle import generate

        temperature = float(self.parameters.get("temperature", "0.2"))
        max_tokens = int(self.parameters.get("max_tokens", "2048"))
        return generate(
            model=self._model_name,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def output(self, filename: str) -> None:
        """Write per-subject and cohort Markdown, JSON, and optional PDF reports."""
        base = Path(filename)
        self._output_base = str(base)
        base.parent.mkdir(parents=True, exist_ok=True)

        meta = {
            "model": self._model_name,
            "hardware": self._hardware_summary,
            "timestamp": self._run_timestamp,
        }

        for subject_id, result in self._results.items():
            subject_base = base.parent / f"{base.name}_{subject_id}"
            md_text = render_subject_report(subject_id, result, meta)
            md_path = subject_base.with_suffix(".md")
            json_path = subject_base.with_suffix(".json")
            md_path.write_text(md_text, encoding="utf-8")
            json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

            if _param_bool(self.parameters, "output_pdf", True):
                pdf_path = subject_base.with_suffix(".pdf")
                try:
                    write_pdf(md_path, pdf_path)
                except Exception as exc:
                    logger.warning("PDF generation failed for %s: %s", subject_id, exc)
                    result.setdefault("warnings", []).append(f"PDF failed: {exc}")

        cohort_base = base.parent / f"{base.name}_cohort"
        cohort_md = render_cohort_report(self._cohort_result, self._results, meta)
        cohort_base.with_suffix(".md").write_text(cohort_md, encoding="utf-8")
        cohort_base.with_suffix(".json").write_text(
            json.dumps(self._cohort_result, indent=2), encoding="utf-8"
        )
        if _param_bool(self.parameters, "output_pdf", True):
            try:
                write_pdf(cohort_base.with_suffix(".md"), cohort_base.with_suffix(".pdf"))
            except Exception as exc:
                logger.warning("Cohort PDF generation failed: %s", exc)
