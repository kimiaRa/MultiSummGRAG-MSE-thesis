# Phase 0 — URL Cleaning & Data Analysis

Phase 0 prepares the raw URL list for the pipeline. It runs in two sequential stages: **URL cleaning** (producing `urls_cleaned.csv`) and **data analysis** (producing quality reports used for thesis documentation).

Run phase 0 with:

```bash
python runs/run_pipeline.py --city cork --phases 0
```

---

## Stage A — URL Cleaning (`url_cleaner.py`)

Reads `data/CITY/urls.csv` (the manually collected URL list) and applies four cleaning steps, writing the result to `data/CITY/urls_cleaned.csv`.

### Step 1 — Deduplication
Exact URL duplicates are dropped using `pandas.drop_duplicates`. Only the first occurrence is kept.

### Step 2 — Facebook URL Removal
URLs containing `facebook` are removed. Facebook pages require a login session and cannot be scraped programmatically — any content behind them is inaccessible to the Phase 1 scraper.

### Step 3 — Dead URL Removal (HTTP 404)
Each URL is checked with an HTTP HEAD request (falling back to GET if HEAD returns 405). URLs that return HTTP 404 are removed. All other status codes — including network errors, timeouts, and server errors — are kept, since a temporary failure does not mean the page is gone.

### Step 4 — Missing Coordinate Resolution
URLs with no latitude/longitude are geocoded in two steps:

1. **Address extraction** — the page is fetched and its text is scanned for a postcode or Eircode using a city-specific regex. If no postcode is found, a broader pattern matches text near the city name.
2. **Nominatim geocoding** — the extracted address is sent to the OpenStreetMap Nominatim API (rate-limited to 1 request/second to comply with usage policy).
3. **City-centre fallback** — if neither step finds a location, the URL is assigned the city's central coordinates and flagged with `coords_approx = True`.

The `coords_approx` flag propagates through all downstream phases and appears as a caveat note in the final report under the Geographic Distribution section.

### City-specific configuration

Cleaning parameters for each city are defined in `CITY_GEOCODING` inside `url_cleaner.py`:

| City | Centre coords | Postcode regex | Fallback keywords |
|---|---|---|---|
| Cork | 51.8985, -8.4756 | Irish Eircode `[A-Z]\d{2}\s?[A-Z0-9]{4}` | Cork |
| Dublin | 53.3498, -6.2603 | Dublin postcode `D\d{1,2}[A-Z]?\s?[A-Z0-9]{4}` | Dublin |
| Brighton & Hove | 50.8225, -0.1372 | UK postcode `BN\d{1,2}\s?\d[A-Z]{2}` | Brighton, Hove, Sussex |

To add a new city, add an entry to `CITY_GEOCODING` in `url_cleaner.py`.

### Output

`data/CITY/urls_cleaned.csv` — four columns:

| Column | Description |
|---|---|
| `URL` | Cleaned, accessible URL |
| `Lat` | Latitude (exact or approximated) |
| `Lng` | Longitude (exact or approximated) |
| `coords_approx` | `True` if coordinates are city-centre fallback |

---

## Stage B — Data Analysis

Runs on the cleaned `urls_cleaned.csv` and produces quality reports for thesis documentation. None of these outputs affect the pipeline — they are for human review only.

### Step 0c — Duplicate detection (`duplicate_detector.py`)
Detects near-duplicate URLs (same domain, different paths) and flags them in the annotated output.

### Step 0d — Non-FSI detection (`fsi_detector.py`)
Uses keyword heuristics to flag URLs that are likely not Food Sharing Initiatives (e.g. news articles, government portal pages). Low-confidence cases are separately flagged for manual review.

### Step 0e — Missing coordinate flagging (`missing_coord_resolver.py`)
Cross-checks coordinate completeness after cleaning and logs any remaining gaps.

### Step 0f — Language detection (`language_detector.py`)
Detects the likely language of each URL's domain. Used to flag non-English content that may affect text extraction quality in Phase 1.

### Step 0g — Statistics & reporting (`stats_reporter.py`, `review_reporter.py`)
Generates summary statistics and a structured review report.

### Outputs

| File | Description |
|---|---|
| `data/CITY/urls_cleaned.csv` | Cleaned URL list used by all subsequent phases |
| `output/phase0_annotated.json` | All URLs with quality flags and stats |
| `output/phase0_review.json` | Flagged URLs requiring manual attention |

---

## Original notebooks

The cleaning logic in `url_cleaner.py` was derived from three exploratory notebooks (`test_urls_cork.ipynb`, `test_urls_dublin.ipynb`, `test_urls_brighton.ipynb`), used to develop and validate the cleaning approach for each city before the logic was consolidated into this reusable script. These notebooks were removed during the final-repository cleanup as pre-consolidation, ad-hoc exploratory artifacts, not part of the Phase 0-5 test suite or final thesis methodology; `url_cleaner.py` (and its coverage in `tests/test_phase0.py`) is the current, active implementation.
