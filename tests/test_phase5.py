import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config_loader import load_city_config

city_name = sys.argv[1] if len(sys.argv) > 1 else "dublin"
cfg       = load_city_config(city_name)

html_path = cfg["report_html"]
pdf_path  = cfg["report_pdf"]

print(f"=== Phase 5 — {cfg['city']}, {cfg['country']} ===\n")

if html_path.exists():
    text = html_path.read_text(encoding="utf-8")
    imgs = text.count("data:image/jpeg;base64")
    svgs = text.count("data:image/svg+xml;base64")
    print(f"✓ HTML report exists")
    print(f"  Size            : {html_path.stat().st_size // 1024} KB")
    print(f"  FSI images      : {imgs}")
    print(f"  Charts (SVG)    : {svgs}")
else:
    print(f"✗ HTML report not found at {html_path}")

if pdf_path.exists():
    print(f"\n✓ PDF report exists")
    print(f"  Size : {pdf_path.stat().st_size // 1024} KB")
else:
    print(f"\n✗ PDF not found at {pdf_path}")

print(f"\n  Open: file://{html_path.resolve()}")