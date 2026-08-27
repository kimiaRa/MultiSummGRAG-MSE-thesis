"""
Re-geocode low-precision coordinates in fsi_records.jsonl.

Reads existing scraped records, tries to extract a more precise address
from the page text/title via Nominatim, and updates any coordinates that
have only 1-2 decimal places of precision (~1 km resolution).

Run:
    python runs/regeocode.py --city dublin
Then re-run phases 2, 4, 5, 6:
    python runs/run_pipeline.py --city dublin --phases 2,4,5,6
"""

import sys, argparse, json, re, time
from pathlib import Path

src = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src))
from config_loader import load_city_config

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

_geo = Nominatim(user_agent="fsi-regeocode-v1")

# Irish Eircode: letter + 2 digits + space + 4 alphanumeric
_EIRCODE = re.compile(r'\b([A-Z]\d{2}\s?[A-Z0-9]{4})\b')
# UK postcode
_UK_POST = re.compile(r'\b([A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2})\b')
# Street address: house number + street name + type
_STREET  = re.compile(
    r'\b(\d{1,4}\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*'
    r'\s+(?:Street|St|Road|Rd|Avenue|Ave|Lane|Drive|Dr'
    r'|Place|Pl|Close|Row|Square|Terrace|Gardens|Park|Crescent|Court))\b'
)


def _decimal_places(val: float) -> int:
    s = f"{val:.10f}".rstrip('0')
    return len(s.split('.')[1]) if '.' in s else 0


def _is_low_precision(geo: dict) -> bool:
    lat, lng = geo.get('lat'), geo.get('lng')
    if lat is None or lng is None:
        return False
    return max(_decimal_places(float(lat)), _decimal_places(float(lng))) <= 2


def _candidates(text: str, title: str, country: str) -> list[str]:
    """Extract candidate address strings to try geocoding."""
    hits = []
    combined = f"{title}\n{text}"

    if "Ireland" in country or "Dublin" in title or "Dublin" in text:
        hits += _EIRCODE.findall(combined)
    else:
        hits += _UK_POST.findall(combined)

    hits += _STREET.findall(combined)

    # title often contains street/place name — try it directly
    if title and len(title.split()) >= 2:
        hits.append(title)

    return list(dict.fromkeys(hits))[:6]  # deduplicate, keep order, cap at 6


def _geocode(query: str, city: str, country_codes: list[str]) -> dict | None:
    for attempt in range(3):
        try:
            loc = _geo.geocode(f"{query}, {city}", country_codes=country_codes, timeout=10)
            time.sleep(1.1)   # Nominatim rate limit: 1 req/s
            if loc:
                return {"lat": loc.latitude, "lng": loc.longitude,
                        "source": "text_extracted", "query": query}
            return None
        except GeocoderTimedOut:
            time.sleep(1.5 ** attempt)
        except GeocoderServiceError:
            time.sleep(2)
    return None


def improve_record(record: dict, city: str, country: str) -> dict | None:
    """Return improved geo dict, or None if no improvement found."""
    text  = record.get("text", "") or ""
    title = record.get("title", "") or ""
    country_codes = ["ie"] if "Ireland" in country else ["gb"]

    for candidate in _candidates(text, title, country):
        result = _geocode(candidate, city, country_codes)
        if result:
            return result

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    args = parser.parse_args()

    cfg = load_city_config(args.city)
    records_path = cfg["fsi_records"]
    records = [json.loads(l) for l in records_path.read_text().splitlines() if l.strip()]

    improved, failed, skipped = 0, 0, 0
    for r in records:
        geo = r.get("geo")
        if not geo or not _is_low_precision(geo):
            skipped += 1
            continue

        orig = (geo.get('lat'), geo.get('lng'))
        result = improve_record(r, cfg["city"], cfg["country"])

        if result:
            old_dp = max(_decimal_places(float(orig[0])), _decimal_places(float(orig[1])))
            new_dp = max(_decimal_places(result['lat']), _decimal_places(result['lng']))
            print(f"  ✓ {r['title'][:50]}")
            print(f"    ({orig[0]}, {orig[1]}) [{old_dp}dp]  →  "
                  f"({result['lat']:.5f}, {result['lng']:.5f}) [{new_dp}dp]"
                  f"  via: {result['query']}")
            r["geo"] = {**geo, **result}
            improved += 1
        else:
            print(f"  ✗ no improvement: {r['title'][:60]}")
            failed += 1

    print(f"\n{'='*55}")
    print(f"  Improved : {improved}")
    print(f"  No match : {failed}")
    print(f"  Skipped  : {skipped} (already precise or no coords)")

    records_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8"
    )
    print(f"  ✓ Updated records → {records_path}")


if __name__ == "__main__":
    main()
