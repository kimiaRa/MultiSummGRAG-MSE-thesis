from pathlib import Path
from jinja2 import Environment, BaseLoader
import base64, json
from chart_renderer import render_bar_chart, render_donut_chart
from map_renderer import render_map


def _to_b64(local_path: str) -> str | None:
    try:
        data = Path(local_path).read_bytes()
        return "data:image/jpeg;base64," + base64.b64encode(data).decode()
    except Exception:
        return None


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>FSI Summary — {{ meta.city }}, {{ meta.country }}</title>
<style>
  :root {
    --green:  #2d6a4f;
    --mid:    #52b788;
    --light:  #f0faf4;
    --border: #d8f3dc;
    --muted:  #666;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: Georgia, serif;
    font-size: 13px;
    line-height: 1.65;
    color: #1b1b1b;
    background: #fff;
    max-width: 860px;
    margin: 0 auto;
    padding: 36px 32px;
  }
  h1 { font-size: 24px; color: var(--green); margin-bottom: 4px; }
  h2 { font-size: 15px; color: var(--green); margin: 24px 0 8px;
       border-bottom: 1.5px solid var(--border); padding-bottom: 4px; }
  p  { margin-bottom: 12px; color: #222; }

  .subtitle { font-size: 11px; color: var(--muted);
              text-transform: uppercase; letter-spacing: 0.05em;
              margin-bottom: 14px; }
  .stat-row { display: flex; gap: 14px; flex-wrap: wrap; margin: 16px 0 20px; }
  .stat-card { background: var(--light); border: 1px solid var(--border);
               border-radius: 8px; padding: 10px 18px;
               text-align: center; min-width: 100px; }
  .stat-card .num { font-size: 22px; font-weight: bold; color: var(--green); }
  .stat-card .lbl { font-size: 10px; color: var(--muted);
                    text-transform: uppercase; letter-spacing: 0.05em; }

  .image-strip { display: flex; gap: 10px; flex-wrap: wrap; margin: 14px 0; }
  .image-strip figure { flex: 1; min-width: 160px; max-width: 210px; margin: 0; }
  .image-strip img { width: 100%; height: 120px; object-fit: cover;
                     border-radius: 6px; border: 1px solid var(--border); }
  .image-strip figcaption { font-size: 10px; color: var(--muted);
                             margin-top: 3px; font-family: sans-serif; }

  .charts-row { display: flex; gap: 20px; margin: 20px 0;
                align-items: flex-start; }
  .chart-box { flex: 1; background: var(--light);
               border: 1px solid var(--border);
               border-radius: 8px; padding: 12px; }
  .chart-box img { width: 100%; }
  .chart-title { font-size: 11px; color: var(--muted);
                 text-transform: uppercase; letter-spacing: 0.05em;
                 margin-bottom: 6px; font-family: sans-serif; }

  .gap-note { background: #fffbf0; border-left: 3px solid #f0b429;
              padding: 8px 12px; font-size: 11px; color: #7a5c00;
              font-family: sans-serif; margin: 16px 0;
              border-radius: 0 4px 4px 0; }

  .map-box { margin: 20px 0 28px; border: 1px solid var(--border);
             border-radius: 8px; overflow: hidden;
             max-width: 680px; }
  .map-box img { width: 100%; display: block; }
  .map-caption { font-size: 10px; color: var(--muted); font-family: sans-serif;
                 padding: 5px 10px; background: var(--light); }

  .footer { margin-top: 36px; padding-top: 10px;
            border-top: 1px solid var(--border);
            font-size: 10px; color: var(--muted); font-family: sans-serif; }

  @media print {
    body { padding: 16px; }
    .charts-row { break-inside: avoid; }
  }
</style>
</head>
<body>

<div class="subtitle">Multimodal Summary Report</div>
<h1>Food Sharing Initiatives (FSIs) in {{ meta.city }}, {{ meta.country }}</h1>



{% if images %}
<div class="image-strip">
  {% for img in images %}
  <figure>
    <img src="{{ img.src }}" alt="{{ img.caption }}">
    <figcaption>{{ img.caption }}{% if img.source_page %}<br><a href="{{ img.source_page }}" target="_blank" style="font-size:9px;word-break:break-all;color:var(--muted);">{{ img.source_page }}</a>{% endif %}</figcaption>
  </figure>
  {% endfor %}
</div>
{% endif %}

<div class="charts-row">
  <div class="chart-box">
    <div class="chart-title">FSIs by district</div>
    <img src="{{ bar_chart }}" alt="FSIs by district">
  </div>
  <div class="chart-box">
    <div class="chart-title">FSI type distribution</div>
    <img src="{{ donut_chart }}" alt="FSI type distribution">
  </div>
</div>

<h2>Geographic Distribution</h2>
{% if map_img %}
<div class="map-box">
  <img src="{{ map_img }}" alt="FSI locations in {{ meta.city }}">
  <div class="map-caption">Figure: Food Sharing Initiative locations across {{ meta.city }}. Districts coloured by FSI density: green = high, yellow = medium, red = low.</div>
</div>
{% endif %}
{% for para in prose.geographic.split('\n\n') %}{% if para.strip() %}<p>{{ para.strip() }}</p>{% endif %}{% endfor %}
{% if facts.approx_coord_count > 0 %}
<div class="gap-note">
  <strong>Location note:</strong> {{ facts.approx_coord_count }} of {{ facts.total }}
  initiatives ({{ facts.approx_coord_pct }}%) were assigned approximate coordinates.
</div>
{% endif %}

<h2>Types of Food Sharing Initiatives</h2>
{% for para in prose.types.split('\n\n') %}{% if para.strip() %}<p>{{ para.strip() }}</p>{% endif %}{% endfor %}

<h2>Operational and Funding Models</h2>
{% for para in prose.operational.split('\n\n') %}{% if para.strip() %}<p>{{ para.strip() }}</p>{% endif %}{% endfor %}

<h2>Reach and Activity</h2>
{% for para in prose.reach.split('\n\n') %}{% if para.strip() %}<p>{{ para.strip() }}</p>{% endif %}{% endfor %}

<h2>Digital Presence and Accessibility</h2>
{% for para in prose.digital.split('\n\n') %}{% if para.strip() %}<p>{{ para.strip() }}</p>{% endif %}{% endfor %}

<h2>Notable Initiatives</h2>
{% for para in prose.notable.split('\n\n') %}{% if para.strip() %}<p>{{ para.strip() }}</p>{% endif %}{% endfor %}

<div class="footer">
  FSI MultiSumm Pipeline &nbsp;·&nbsp; {{ meta.city }}, {{ meta.country }}
  &nbsp;·&nbsp; {{ facts.total }} initiatives across {{ facts.district_count }} districts
</div>

</body>
</html>"""


def render_html(schema: dict, facts: dict, prose: dict,
                images: list[dict], records: list[dict] = None,
                cfg: dict = None) -> str:
    embedded_images = []
    for img in images:
        src = _to_b64(img.get("local_path", ""))
        if src:
            embedded_images.append({**img, "src": src})

    bar_chart   = render_bar_chart(facts["district_counts"])
    donut_chart = render_donut_chart(facts["type_counts"])

    map_img = None
    if records and cfg:
        print("  Rendering FSI map...")
        map_img = render_map(records, cfg)

    env      = Environment(loader=BaseLoader())
    template = env.from_string(TEMPLATE)
    return template.render(
        meta        = schema["meta"],
        facts       = facts,
        prose       = prose,
        images      = embedded_images,
        bar_chart   = bar_chart,
        donut_chart = donut_chart,
        map_img     = map_img,
    )


def save_html(html: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    print(f"✓ HTML report saved to {path}")