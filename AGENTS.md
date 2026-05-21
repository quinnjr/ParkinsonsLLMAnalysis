# AGENTS.md — ParkinsonsLLMAnalysis

## Overview

Standalone PluMA Python plugin for hardware-aware local LLM analysis of Parkinson's pipeline outputs. Ingests upstream CSV/JSON/TXT artifacts into modality-scoped ChromaDB collections, runs RAG-augmented scientific and clinical analysis via Ollama (Gemma 4 tiers with fallback), writes per-subject Markdown/JSON/PDF reports with PD likelihood assessments, then shuts down Ollama and cleans temp artifacts if configured.

## Architecture

```
ParkinsonsLLMAnalysis.py   — PluMA input/run/output orchestrator
hardware.py                — CPU/RAM/GPU detection
model_catalog.py           — Gemma 4 tier table (HF provenance → Ollama tags)
ollama_lifecycle.py        — start/stop/pull/generate with ownership tracking
artifact_ingest.py         — pipeline CSV/JSON → DocumentRecord
chroma_store.py            — ChromaDB build + subject-filtered RAG queries
prompts.py                 — clinical/scientific/PD likelihood prompt templates
analysis.py                — per-subject multi-pass RAG + LLM orchestration
report_writer.py           — Markdown report assembly
pdf_writer.py              — Markdown → HTML → Micropdf PDF
cleanup.py                 — temp artifact removal + Ollama shutdown
```

## Plugin contract

- Class: `ParkinsonsLLMAnalysis` in `ParkinsonsLLMAnalysis.py`
- Loader wrapper: `ParkinsonsLLMAnalysisPlugin` in `ParkinsonsLLMAnalysisPlugin.py`
- Parameters: whitespace key-value (`parameters.parkinsonsllmanalysis.txt`)
- Outputs: `{base}_{subject_id}.{md,json,pdf}` plus `{base}_cohort.{md,json,pdf}`

## ChromaDB collections

| Collection | work_type |
|---|---|
| microbiome | microbiome |
| proteomics | proteomics |
| clinical_assay | clinical_assay |
| clinical_labels | clinical |
| integration_mofa | integration_mofa |
| integration_snf | integration_snf |
| integration_fusion | integration_fusion |
| explainability | explainability |
| ml_predictions | ml_predictions |
| stage1_outputs | stage1 |

## Model selection

Hardware detection selects the highest Gemma 4 Ollama tier that fits VRAM/RAM. Requires Ollama ≥0.20 for Gemma 4. Falls back to `llama3.1:8b`.

## Prerequisites

1. Ollama ≥0.20 — https://ollama.com
2. Optional: Micropdf Python bindings for PDF output
3. PluMA shared `.venv` installs deps from `requirements.txt` during `scons`

## Testing

```bash
pip install -r requirements.txt -r requirements-test.txt
pytest -v
```

Integration tests requiring live Ollama are marked `@pytest.mark.integration`.

## Attribution

Author: Joseph R. Quinn <quinn.josephr@protonmail.com>
License: MIT
