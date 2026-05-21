"""
Markdown to PDF conversion via Micropdf.

Author: Joseph R. Quinn <quinn.josephr@protonmail.com>
License: MIT
"""

from __future__ import annotations

from pathlib import Path

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
body {{ font-family: Georgia, serif; margin: 2cm; line-height: 1.5; color: #222; }}
h1 {{ color: #1a365d; border-bottom: 2px solid #1a365d; padding-bottom: 0.3em; }}
h2 {{ color: #2c5282; margin-top: 1.5em; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
th, td {{ border: 1px solid #ccc; padding: 0.4em 0.6em; text-align: left; }}
th {{ background: #edf2f7; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def markdown_to_html(md_text: str) -> str:
    """Convert Markdown to styled HTML."""
    import markdown

    body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
    return HTML_TEMPLATE.format(body=body)


def write_pdf(md_path: str | Path, pdf_path: str | Path) -> None:
    """Convert Markdown file to PDF using Micropdf."""
    md_path = Path(md_path)
    pdf_path = Path(pdf_path)
    html = markdown_to_html(md_path.read_text(encoding="utf-8"))

    html_path = pdf_path.with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")

    try:
        from micropdf import html_to_pdf
    except ImportError as exc:
        raise RuntimeError(
            "micropdf is required for PDF output. Install micropdf Python bindings."
        ) from exc

    html_to_pdf(html, str(pdf_path))
