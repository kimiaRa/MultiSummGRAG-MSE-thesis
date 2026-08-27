import yaml
import geopandas as gpd
from pathlib import Path
from shapely.geometry import box

ROOT_DIR = Path(__file__).parent.parent


def load_city_config(city_name: str) -> dict:
    yaml_path = ROOT_DIR / "config" / f"{city_name}.yaml"
    if not yaml_path.exists():
        available = [p.stem for p in (ROOT_DIR / "config").glob("*.yaml")]
        raise FileNotFoundError(
            f"No config found for '{city_name}'. "
            f"Available: {available}"
        )

    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_dir = ROOT_DIR / cfg["data_dir"]
    out_dir  = data_dir / "output"
    slug     = city_name.lower().replace(" ", "_").replace("&", "and")

    cfg["root_dir"]       = ROOT_DIR
    cfg["data_dir"]       = data_dir
    cfg["out_dir"]        = out_dir
    cfg["raw_dir"]        = data_dir / "raw"
    cfg["images_dir"]     = out_dir / "images"
    cfg["urls_file"]      = ROOT_DIR / cfg["urls_file"]
    cfg["raw_urls_file"] = ROOT_DIR / cfg.get("raw_urls_file",
                                              cfg["data_dir"] / "raw_urls.csv")
    cfg["districts_file"] = ROOT_DIR / cfg["districts_file"]

    cfg["urls_raw"]        = data_dir / "urls.csv"    # raw URL list — input to phase 0 cleaning
    cfg["phase0_urls_out"] = data_dir / "urls.csv"    # phase 0 write target — never urls_cleaned.csv

    cfg["fsi_records"]    = out_dir / "fsi_records.jsonl"
    cfg["fsi_enriched"]   = out_dir / "fsi_enriched.jsonl"
    cfg["graph_dir"]      = data_dir / "graphrag"
    cfg["phase3_answers"] = out_dir / "phase3_answers.json"
    cfg["phase4_schema"]  = out_dir / "phase4_schema.json"
    cfg["report_html"]    = out_dir / f"report_{slug}.html"
    cfg["report_pdf"]     = out_dir / f"report_{slug}.pdf"
    cfg["eval_json"]      = out_dir / f"eval_{slug}.json"
    cfg["eval_html"]      = out_dir / f"eval_{slug}.html"

    cfg["chatgpt_baseline_html"] = out_dir / f"report_{slug}_chatgpt.html"
    cfg["eval_chatgpt_json"]     = out_dir / f"eval_{slug}_chatgpt.json"
    cfg["eval_chatgpt_html"]     = out_dir / f"eval_{slug}_chatgpt.html"

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (data_dir / "raw").mkdir(parents=True, exist_ok=True)

    return cfg


def _fetch_osm_district(name: str, city: str, bbox_geom) -> object | None:
    """
    Try to fetch a real OSM polygon for a district.
    Returns the geometry only if it is a polygon that:
      - intersects the YAML bbox
      - has an area between 5% and 500% of the YAML bbox area
        (rejects both point-like results and city-wide blobs)
    Otherwise returns None so the caller falls back to the bounding box.
    """
    try:
        import osmnx as ox
        gdf = ox.geocode_to_gdf(f"{name}, {city}")
        geom = gdf.geometry.iloc[0]
        if geom.geom_type not in ("Polygon", "MultiPolygon"):
            return None
        if not geom.intersects(bbox_geom):
            return None
        bbox_area = bbox_geom.area
        osm_area  = geom.area
        if bbox_area > 0 and not (0.05 * bbox_area <= osm_area <= 5 * bbox_area):
            return None
        return geom
    except Exception:
        pass
    return None


def build_districts_geojson(cfg: dict):
    out_path = Path(cfg["districts_file"])
    if out_path.exists():
        return

    districts_raw = cfg.get("districts", {})
    if not districts_raw:
        raise ValueError(
            f"No districts defined in config for {cfg['city']} "
            f"and no districts.geojson found at {out_path}"
        )

    rows = []
    osm_count = 0
    for name, bbox in districts_raw.items():
        min_lng, min_lat, max_lng, max_lat = bbox
        bbox_geom = box(min_lng, min_lat, max_lng, max_lat)

        osm_geom = _fetch_osm_district(name, cfg["city"], bbox_geom)
        if osm_geom is not None:
            geometry = osm_geom
            source   = "osm"
            osm_count += 1
        else:
            geometry = bbox_geom
            source   = "bbox"

        rows.append({
            "name":          name,
            "district_code": name,
            "geometry":      geometry,
            "source":        source,
        })

    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GeoJSON")
    bbox_count = len(rows) - osm_count
    print(f"  ✓ Built {len(gdf)} district boundaries → {out_path}")
    print(f"    ({osm_count} from OSM polygons, {bbox_count} from bounding boxes)")

# ── Shared pipeline constants ─────────────────────────────────────────────────

SCRAPER = {
    "timeout_ms":   20_000,
    "wait_until":   "networkidle",
    "max_retries":  3,
    "concurrency":  3,
    "max_depth":    1,   # 0 = seed only; 1 = seed + one level of internal links
    "max_subpages": 5,   # max internal links to follow per seed page
}

IMAGE = {
    "min_width":        200,
    "min_height":       200,
    "max_per_page":     10,
    "hash_threshold":   8,
    "prefer_landscape": True,
}

LLM = {
    "model":      "qwen3:14b",
    "max_tokens": 256,
    "batch_size": 1,
}

OLLAMA = {
    "llm_model":   "llama3.2-graphrag",
    "embed_model": "nomic-embed-text",
    "embed_dim":   768,
    "max_async":   2,
    "model":       "qwen3:14b",
    "num_ctx":     16384,
}

# Phase 6 judge model — deliberately different from OLLAMA["model"]/LLM["model"],
# which is what Phase 5 uses to *write* the report. Using the same model as both
# generator and judge is a documented self-preference bias in LLM-as-judge setups;
# keeping them distinct is required for the judgments to be an independent check.
JUDGE_MODEL = {
    "model":   "qwen2.5:7b",
    "num_ctx": 16384,
}

# Number of repeated G-Eval trials per report (same prompt, temperature 0).
# LLM-as-judge scores vary across identical repeated runs even at temperature
# 0 ("trial bias", Zeng et al. 2025, arXiv:2506.06331); averaging over
# N_GEVAL_TRIALS trials and recording their spread makes that variance visible
# instead of silently trusting a single sample.
N_GEVAL_TRIALS = 3

POPULARITY = {
    "high":   ["hundreds", "thousands", "weekly", "daily", "large community",
               "well established", "award", "funded", "network"],
    "medium": ["monthly", "regular", "growing", "volunteers", "members", "active"],
    "low":    ["new", "small", "pilot", "occasional", "starting"],
}

# Headline final_score = weighted sum of these two independent, orthogonal
# pillars only: a deterministic factual-accuracy check and a validated LLM
# judgment of analytical quality. Everything else Phase 6 computes (content
# coverage, informativeness, visual relevance, coherence, rouge) is reported
# as a diagnostic, not blended into final_score — see aggregator.py.
PRIMARY_WEIGHTS = {
    "factual_precision": 0.5,
    "geval":             0.5,
}

REQUIRED_SECTIONS = [
    "Geographic Distribution",
    "Types of Food Sharing Initiatives",
    "Operational and Funding Models",
    "Reach and Activity",
    "Digital Presence and Accessibility",
    "Notable Initiatives",
]

SUMMARY_QUERIES = [
    {
        "field": "geographic_distribution",
        "mode":  "global",
        "query": "What is the geographic distribution of Food Sharing Initiatives across city districts? Which districts have the most number of Food Sharing Initiatives?",
    },
    {
        "field": "fsi_types",
        "mode":  "global",
        "query": "What types of Food Sharing Initiatives exist in the city? Describe each type and give examples.",
    },
    {
        "field": "operational_levels",
        "mode":  "global",
        "query": "How are Food Sharing Initiatives funded and operated? Which are government-funded, community-led, or NGO-led?",
    },
    {
        "field": "popularity",
        "mode":  "global",
        "query": "Which Food Sharing Initiatives are the most active and popular? What signals indicate their reach and activity level?",
    },
    # {
    #     "field": "sentiment",
    #     "mode":  "global",
    #     "query": "What is the overall public sentiment toward Food Sharing Initiatives? Are there any negative experiences or concerns mentioned?",
    # },
    {
        "field": "district_summaries",
        "mode":  "local",
        "query": "Summarise the Food Sharing Initiatives in each city district, including their types and operational models.",
    },
    {
        "field": "notable_initiatives",
        "mode":  "local",
        "query": "Which individual Food Sharing Initiatives stand out as particularly prominent or impactful?",
    },
]