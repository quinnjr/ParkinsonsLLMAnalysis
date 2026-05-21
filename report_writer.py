"""
Markdown report generation.

Author: Joseph R. Quinn <quinn.josephr@protonmail.com>
License: MIT
"""

from __future__ import annotations

from typing import Any

MODALITY_TITLES = {
    "microbiome": "Microbiome Findings",
    "proteomics": "Proteomics and Alpha-Synuclein Findings",
    "integration_mofa": "MOFA+ Integration Results",
    "integration_snf": "SNF Integration Results",
    "integration_fusion": "Early Fusion Results",
    "explainability": "SHAP Explainability Highlights",
}


def render_subject_report(
    subject_id: str,
    result: dict[str, Any],
    meta: dict[str, str],
) -> str:
    """Render full Markdown report for one subject."""
    pd = result.get("pd_likelihood", {})
    lines = [
        f"# Parkinson's Pipeline Analysis Report — {subject_id}",
        "",
        "## Run Metadata",
        f"- **Model:** {meta.get('model', result.get('model', 'unknown'))}",
        f"- **Hardware:** {meta.get('hardware', 'unknown')}",
        f"- **Timestamp:** {meta.get('timestamp', 'unknown')}",
        "",
        "## Subject Profile",
        f"- **Subject ID:** {subject_id}",
        f"- **RAG chunks used:** {result.get('rag_chunk_count', 0)}",
        "",
        "## Parkinson's Likelihood Assessment",
        f"- **Label:** {pd.get('label', 'indeterminate')}",
        f"- **Confidence:** {pd.get('confidence', 'N/A')}",
        f"- **Basis:** {pd.get('model_basis', 'N/A')}",
        "",
    ]

    if pd.get("narrative"):
        lines.extend([pd["narrative"], ""])

    if pd.get("supporting_evidence"):
        lines.append("**Supporting evidence:**")
        for item in pd["supporting_evidence"]:
            lines.append(f"- {item}")
        lines.append("")

    if pd.get("contradicting_evidence"):
        lines.append("**Contradicting evidence:**")
        for item in pd["contradicting_evidence"]:
            lines.append(f"- {item}")
        lines.append("")

    modality_sections = result.get("modality_sections", {})
    for work_type, title in MODALITY_TITLES.items():
        lines.extend([f"## {title}", modality_sections.get(work_type, "No data available."), ""])

    lines.extend(
        [
            "## Scientific Analysis",
            result.get("scientific_analysis", "Not available."),
            "",
            "## Clinical Analysis",
            result.get("clinical_analysis", "Not available."),
            "",
            "## Data Limitations",
            "This report is generated from available pipeline artifacts indexed at run time. "
            "Missing modalities, incomplete subject linkage, or classifier outputs without "
            "sample IDs reduce certainty. Findings are for research interpretation only and "
            "are not a clinical diagnosis.",
            "",
        ]
    )
    return "\n".join(lines)


def render_cohort_report(
    cohort_result: dict[str, Any],
    subject_results: dict[str, dict[str, Any]],
    meta: dict[str, str],
) -> str:
    """Render cohort-level rollup Markdown."""
    lines = [
        "# Parkinson's Pipeline Cohort Analysis",
        "",
        "## Run Metadata",
        f"- **Model:** {meta.get('model', 'unknown')}",
        f"- **Hardware:** {meta.get('hardware', 'unknown')}",
        f"- **Timestamp:** {meta.get('timestamp', 'unknown')}",
        f"- **Subjects analyzed:** {cohort_result.get('subject_count', 0)}",
        "",
        "## Per-Subject PD Likelihood Summary",
        "",
        "| Subject | Label | Confidence |",
        "| --- | --- | --- |",
    ]

    for sid, result in subject_results.items():
        pd = result.get("pd_likelihood", {})
        lines.append(
            f"| {sid} | {pd.get('label', 'indeterminate')} | {pd.get('confidence', 'N/A')} |"
        )

    lines.extend(["", "## Cohort Notes", ""])
    for sid, result in subject_results.items():
        pd = result.get("pd_likelihood", {})
        narrative = pd.get("narrative", "")
        if narrative:
            lines.append(f"**{sid}:** {narrative}")
            lines.append("")

    return "\n".join(lines)
