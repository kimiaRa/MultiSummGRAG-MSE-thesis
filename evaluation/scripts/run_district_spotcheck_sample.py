#!/usr/bin/env python3
"""
Pre-fill evaluation/results/district_spotcheck.csv with a random sample for
manual address-vs-district spot-checking. Read-only apart from that one
file (an existing 15-row template -- item_id/city columns already fixed at
3 rows per city in order barcelona/brighton/dublin/london/milan -- only
url/initiative_name/pipeline_lat/pipeline_lng/pipeline_district are filled
in; address_on_page/district_from_address/MATCH/NOTE are left blank for the
user to fill by hand after reading the pages).

Sampling frame per city: URLs that BOTH (a) appear in urls_cleaned.csv
(Phase-0 cleaning kept them) AND (b) are marked "Retained" in that city's
baseline_v2 HTML screening ledger. Matched by exact URL string (not
normalised) -- checked directly: 1 of Barcelona's 47 retained-ledger URLs
(https://www.nutricionsinfronteras.org/home) has no match in
urls_cleaned.csv even under normalisation (it's a different subdomain,
ca.nutricionsinfronteras.org, with a different path -- a genuinely
different URL, not a formatting artefact), so it's correctly excluded.
Qualifying-set sizes: barcelona 46, brighton 24, dublin 36, london 33,
milan 34 (all cleaned rows recovered for every city except barcelona's one
real non-match above).

Sampling: seed=42, a single random.Random(42) instance consumed in a fixed
city order (barcelona, brighton, dublin, london, milan -- same CITIES order
used throughout evaluation/*.py) so the draw is exactly reproducible.
Dublin is sampled specially -- see below.

Dublin bias: pipeline_district is computed for every qualifying Dublin URL
FIRST (point-in-polygon, same method as the rest of this file), then the
subset assigned to "Dublin 2" is sampled from before the general pool, per
instruction ("that district holds 15 of 69 initiatives"). If fewer than 3
Dublin 2 rows qualify, the shortfall is filled from the remaining
qualifying pool (Dublin 2 rows excluded, so no double-draw).

pipeline_district: point-in-polygon (shapely contains/touches) of
(pipeline_lat, pipeline_lng) against data/<city>/districts.geojson -- the
same ground-truth recompute method e1_evidence_consistency.py's
recompute_district_membership() uses, applied per-URL here instead of as
an aggregate count. "outside all polygons" if the point matches no
district.

Run from the repo root with the project venv active:
    venv/bin/python3 evaluation/scripts/run_district_spotcheck_sample.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from bs4 import BeautifulSoup
from shapely.geometry import Point

_HERE = Path(__file__).resolve().parent
_EVAL_DIR = _HERE.parent
sys.path.insert(0, str(_EVAL_DIR))
import e1_evidence_consistency as e1  # noqa: E402

DATA = e1.DATA
CITIES = e1.CITIES
SPOTCHECK_CSV = _EVAL_DIR / "results" / "district_spotcheck.csv"
SEED = 42


def parse_ledger_html(city: str) -> pd.DataFrame:
    html = (DATA / city / "output" / f"report_{city}_baseline_v2.html").read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")
    tbody = soup.find("table").find("tbody")
    rows = []
    for tr in tbody.find_all("tr"):
        tds = tr.find_all("td")
        a = tds[1].find("a") if len(tds) > 1 else None
        url = (a.get("href") if a else tds[1].get_text(strip=True)) if len(tds) > 1 else None
        decision = tds[2].get_text(strip=True) if len(tds) > 2 else None
        rows.append({"url": url, "decision": decision})
    return pd.DataFrame(rows)


def load_qualifying(city: str) -> pd.DataFrame:
    """cleaned URLs (with lat/lng) that are also Retained in the baseline_v2
    ledger, exact-URL-matched."""
    cleaned = pd.read_csv(DATA / city / "urls_cleaned.csv")
    ledger = parse_ledger_html(city)
    retained_urls = set(ledger[ledger.decision == "Retained"]["url"])
    qual = cleaned[cleaned["URL"].isin(retained_urls)].copy()
    qual = qual.rename(columns={"URL": "url", "Lat": "lat", "Lng": "lng"})
    return qual[["url", "lat", "lng"]].reset_index(drop=True)


def assign_district(lat: float, lng: float, gdf: gpd.GeoDataFrame) -> str:
    if pd.isna(lat) or pd.isna(lng):
        return "outside all polygons"
    pt = Point(float(lng), float(lat))
    for _, drow in gdf.iterrows():
        if drow.geometry.contains(pt) or drow.geometry.touches(pt):
            return str(drow.get("name"))
    return "outside all polygons"


def load_initiative_names(city: str) -> dict[str, str]:
    """url -> initiative_name from fsi_enriched.jsonl, IF such a field
    exists. Checked directly across all 5 cities' fsi_enriched.jsonl: the
    only per-record keys are url/title/text/location/fsi_type/
    operational_level/popularity/images/description/scraped_at/error/phase
    -- there is no 'name' or 'initiative_name' field anywhere (only
    'title', the raw scraped <title> tag text, e.g. often containing
    site-branding suffixes -- not a clean initiative name, so NOT used as
    a substitute here). Returns {} for every city; kept as a real lookup
    (not hardcoded blank) so this starts working automatically if such a
    field is ever added."""
    import json
    path = DATA / city / "output" / "fsi_enriched.jsonl"
    names: dict[str, str] = {}
    if not path.exists():
        return names
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            name = rec.get("initiative_name") or rec.get("name")
            if name:
                names[rec.get("url")] = name
    return names


def main() -> None:
    rng = random.Random(SEED)
    samples: dict[str, pd.DataFrame] = {}

    for city in CITIES:
        qual = load_qualifying(city)
        gdf = gpd.read_file(DATA / city / "districts.geojson")
        qual["pipeline_district"] = qual.apply(
            lambda r: assign_district(r["lat"], r["lng"], gdf), axis=1)

        if city == "dublin":
            dublin2 = qual[qual["pipeline_district"] == "Dublin 2"]
            rest = qual[qual["pipeline_district"] != "Dublin 2"]
            n_d2 = len(dublin2)
            take_d2 = min(3, n_d2)
            picked_d2 = dublin2.sample(n=take_d2, random_state=rng.randint(0, 2**31 - 1)) if take_d2 else dublin2.iloc[0:0]
            remaining_needed = 3 - take_d2
            picked_rest = rest.sample(n=remaining_needed, random_state=rng.randint(0, 2**31 - 1)) if remaining_needed else rest.iloc[0:0]
            picked = pd.concat([picked_d2, picked_rest], ignore_index=True)
            print(f"dublin: {n_d2} of {len(qual)} qualifying URLs are Dublin 2 -- "
                  f"took {take_d2} from Dublin 2, {remaining_needed} at random from the rest")
        else:
            picked = qual.sample(n=3, random_state=rng.randint(0, 2**31 - 1)).reset_index(drop=True)

        names = load_initiative_names(city)
        picked["initiative_name"] = picked["url"].map(names).fillna("")
        samples[city] = picked

    # ── fill the existing template, preserving item_id/city/row order ───
    spot = pd.read_csv(SPOTCHECK_CSV, dtype=str, keep_default_na=False)
    city_counters: dict[str, int] = {c: 0 for c in CITIES}
    for idx, row in spot.iterrows():
        city = row["city"]
        i = city_counters[city]
        picked = samples[city]
        r = picked.iloc[i]
        spot.at[idx, "url"] = str(r["url"])
        spot.at[idx, "initiative_name"] = str(r["initiative_name"])
        spot.at[idx, "pipeline_lat"] = str(r["lat"])
        spot.at[idx, "pipeline_lng"] = str(r["lng"])
        spot.at[idx, "pipeline_district"] = str(r["pipeline_district"])
        city_counters[city] += 1
    spot.to_csv(SPOTCHECK_CSV, index=False)
    print(f"\nWrote {SPOTCHECK_CSV}")
    print(spot.to_string(index=False))

    # ── coordinate collapse: how many of a city's cleaned URLs share a
    # coordinate with at least one SAMPLED row ─────────────────────────
    print("\n" + "=" * 78)
    print("Coordinate collapse per city (cleaned URLs sharing a coordinate with >=1 sampled row)")
    print("=" * 78)
    for city in CITIES:
        cleaned = pd.read_csv(DATA / city / "urls_cleaned.csv")
        cleaned_coords = list(zip(cleaned["Lat"].round(6), cleaned["Lng"].round(6)))
        sampled_coords = set(
            (round(float(lat), 6), round(float(lng), 6))
            for lat, lng in zip(samples[city]["lat"], samples[city]["lng"])
        )
        n_sharing = sum(1 for c in cleaned_coords if c in sampled_coords)
        print(f"{city}: {n_sharing} of {len(cleaned)} cleaned URLs share a coordinate with "
              f"one of the {len(samples[city])} sampled rows (sampled coords: {sorted(sampled_coords)})")


if __name__ == "__main__":
    main()
