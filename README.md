# ParkinsonsLLMAnalysis

PluMA plugin for RAG-augmented local LLM analysis of Parkinson's disease multi-omics pipeline outputs.

## Features

- Hardware-aware model selection (Gemma 4 tiers via Ollama, with Llama 3.1 fallback)
- Ingests all configured pipeline artifacts into modality-scoped ChromaDB collections
- Per-subject scientific and clinical analysis with PD likelihood assessment
- Markdown + JSON + PDF reports (PDF via Micropdf)
- Ollama lifecycle management (auto-start/stop with ownership tracking)
- Temp artifact cleanup after each run

## Prerequisites

- **Ollama ≥0.20** (required for Gemma 4) — [ollama.com](https://ollama.com)
- **Micropdf** Python bindings (optional, for PDF output)
- PluMA with Python plugin support

## Quick start

```bash
# Symlink into PluMA plugins
cd PluMA/plugins
ln -sfn ../../ParkinsonsLLMAnalysis ParkinsonsLLMAnalysis

# Run example (requires Ollama)
./pluma plugins/ParkinsonsLLMAnalysis/example/config.txt
```

## Parameters

See [`parameters.parkinsonsllmanalysis.txt`](parameters.parkinsonsllmanalysis.txt).

Key parameters:

| Parameter | Description |
|---|---|
| `subjects_file` | Sample manifest (required) |
| `work_dir` | Temp workspace for ChromaDB (required) |
| `syn_features` | Merged α-syn feature matrix |
| `stage1_json_dir` | Directory of Stage1Output JSON files |
| `model_name` | Ollama tag or `auto` |
| `ollama_auto_start` | Start Ollama if not running |
| `ollama_auto_stop` | Stop Ollama if we started it |
| `output_pdf` | Generate PDF reports |

## Outputs

For each subject: `{output}_{subject_id}.md`, `.json`, `.pdf`

Cohort rollup: `{output}_cohort.md`, `.json`, `.pdf`

## Testing

```bash
pip install -r requirements.txt -r requirements-test.txt
pytest -v
```

## License

MIT — Joseph R. Quinn <quinn.josephr@protonmail.com>
