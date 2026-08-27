import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "phase_1"))

import asyncio, json
from tqdm import tqdm
from config import CITIES
from scraper import scrape_city, load_urls
from merger import build_fsi_record


async def run(city_name: str):
    from config_loader import load_city_config, build_districts_geojson     
    cfg = load_city_config(city_name)
    out_dir = Path(cfg["out_dir"])
    img_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{city_name}] Loading URLs...")
    rows = load_urls(cfg["urls_file"])
    print(f"[{city_name}] Scraping {len(rows)} URLs...")

    scraped_results = await scrape_city(cfg, rows)

    seen_hashes = set()
    records     = []
    for scraped in tqdm(scraped_results, desc="Building FSI records"):
        record = build_fsi_record(scraped, city_name, img_dir, seen_hashes)
        records.append(record)

    out_path = Path(cfg["out_dir"]) / "fsi_records.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok   = sum(1 for r in records if r["error"] is None)
    geo  = sum(1 for r in records if r["geo"] is not None)
    imgs = sum(len(r["images"]) for r in records)

    print(f"\n✓ {ok}/{len(records)} pages scraped successfully")
    print(f"✓ {geo}/{len(records)} FSIs geocoded")
    print(f"✓ {imgs} images saved")
    print(f"✓ Output → {out_path}")


if __name__ == "__main__":
    import sys as _sys
    city = _sys.argv[1] if len(_sys.argv) > 1 else "dublin"
    run(city)