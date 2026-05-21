import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pdf_writer import markdown_to_html


def test_markdown_to_html_contains_title():
    html = markdown_to_html("# Report Title\n\nBody text.")
    assert "<h1>Report Title</h1>" in html
    assert "Georgia" in html


def test_write_pdf_calls_micropdf(tmp_path, monkeypatch):
    import pdf_writer

    mock_html_to_pdf = MagicMock()
    fake_micropdf = types.SimpleNamespace(html_to_pdf=mock_html_to_pdf)
    monkeypatch.setitem(sys.modules, "micropdf", fake_micropdf)

    md_path = tmp_path / "report.md"
    pdf_path = tmp_path / "report.pdf"
    md_path.write_text("# Report Title\n\nContent.", encoding="utf-8")

    pdf_writer.write_pdf(md_path, pdf_path)

    mock_html_to_pdf.assert_called_once()
    call_html = mock_html_to_pdf.call_args[0][0]
    assert "Report Title" in call_html
