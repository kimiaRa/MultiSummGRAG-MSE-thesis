from pathlib import Path


def export_pdf(html_path: Path, pdf_path: Path):
    try:
        from weasyprint import HTML
        HTML(filename=str(html_path)).write_pdf(str(pdf_path))
        print(f"✓ PDF report saved to {pdf_path}")
    except ImportError:
        print("WeasyPrint not installed — skipping PDF export")
        print("  Run: pip install weasyprint")
    except Exception as e:
        print(f"  PDF export failed: {e}")
        print("  The HTML report is still valid and printable from a browser")