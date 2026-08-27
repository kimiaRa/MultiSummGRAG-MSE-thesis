import re
import time
import requests
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter


# ── City-specific geocoding configuration ─────────────────────────────────────
# To add a new city, add an entry here keyed by the city name in the YAML.

CITY_GEOCODING = {
    "Cork": {
        "center":         (51.8985, -8.4756),
        "postcode_re":    r'[A-Z]\d{2}\s?[A-Z0-9]{4}',
        "text_keywords":  ["Cork"],
        "geocode_suffix": "Cork, Ireland",
    },
    "Dublin": {
        "center":         (53.3498, -6.2603),
        "postcode_re":    r'\bD\d{1,2}[A-Z]?\s?[A-Z0-9]{4}\b',
        "text_keywords":  ["Dublin"],
        "geocode_suffix": "Dublin, Ireland",
    },
    "Brighton & Hove": {
        "center":         (50.8225, -0.1372),
        "postcode_re":    r'BN\d{1,2}\s?\d[A-Z]{2}',
        "text_keywords":  ["Brighton", "Hove", "Sussex"],
        "geocode_suffix": "Brighton, UK",
    },
    "London": {
        "center":         (51.5074, -0.1278),
        "postcode_re":    r'[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}',
        "text_keywords":  ["London"],
        "geocode_suffix": "London, UK",
    },
    "Barcelona": {
        "center":         (41.3851, 2.1734),
        "postcode_re":    r'\b08\d{3}\b',
        "text_keywords":  ["Barcelona", "Barcelone"],
        "geocode_suffix": "Barcelona, Spain",
    },
    "Milan": {
        "center":         (45.4642, 9.1900),
        "postcode_re":    r'\b20\d{3}\b',
        "text_keywords":  ["Milano", "Milan"],
        "geocode_suffix": "Milan, Italy",
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_accessible(url: str, timeout: int = 10) -> bool:
    """
    Returns False for confirmed-dead URLs: 404, DNS failure, SSL errors,
    connection refused. Returns True for timeouts and other transient errors.
    """
    try:
        r = requests.head(url, timeout=timeout, allow_redirects=True,
                          headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 405:
            r = requests.get(url, timeout=timeout, allow_redirects=True,
                             headers={"User-Agent": "Mozilla/5.0"})
        return r.status_code != 404
    except requests.exceptions.SSLError:
        return False        # bad/expired/mismatched certificate — permanent
    except requests.exceptions.ConnectionError as e:
        msg = str(e).lower()
        if any(s in msg for s in ("name or service not known", "nodename nor servname",
                                   "connection refused", "err_name_not_resolved",
                                   "getaddrinfo failed", "no address associated")):
            return False    # DNS failure or port closed — permanent
        return True         # other connection error — might be transient
    except requests.exceptions.Timeout:
        return True         # slow site — keep it, scraper has its own timeout
    except Exception:
        return True         # unknown error — keep to be safe


def _scrape_address(url: str, postcode_re: str,
                    keywords: list[str], timeout: int = 10) -> str | None:
    """Tries to extract a postcode or city-adjacent phrase from the page text."""
    try:
        r = requests.get(url, timeout=timeout,
                         headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")
        text = soup.get_text(" ", strip=True)

        match = re.search(postcode_re, text)
        if match:
            return match.group(0)

        kw_pattern = r'(.{0,60}(?:' + "|".join(re.escape(k) for k in keywords) + r').{0,60})'
        match = re.search(kw_pattern, text)
        return match.group(1).strip() if match else None
    except Exception:
        return None


# ── Main entry point ──────────────────────────────────────────────────────────

def clean_urls(city: str, raw_path: Path, out_path: Path) -> pd.DataFrame:
    """
    Cleans a raw URL CSV for the given city and writes urls_cleaned.csv.

    Steps:
      1. Remove duplicate URLs
      2. Remove Facebook URLs (require login, inaccessible to scrapers)
      3. Remove dead URLs (HTTP 404 confirmed)
      4. Fill missing coordinates via page scraping + Nominatim geocoding,
         falling back to city-centre coords flagged as coords_approx=True
    """
    geo_cfg = CITY_GEOCODING.get(city)
    if not geo_cfg:
        raise ValueError(
            f"No geocoding config for '{city}'. "
            f"Add it to CITY_GEOCODING in url_cleaner.py."
        )

    center_lat, center_lng = geo_cfg["center"]

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"  Loading {raw_path} ...")
    df = pd.read_csv(raw_path)
    df = df.rename(columns={"Lon": "Lng"})   # normalise London-style column name
    print(f"  {len(df)} URLs loaded")

    # ── Step 1: Deduplicate ───────────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["URL"]).reset_index(drop=True)
    if len(df) < before:
        print(f"  Step 1 — {before - len(df)} duplicates removed → {len(df)} URLs")

    # ── Step 2: Remove login-walled social media URLs ────────────────────────
    _blocked_domains = ["facebook.com", "instagram.com", "tiktok.com", "twitter.com", "x.com"]
    before = len(df)
    pattern = "|".join(_blocked_domains)
    df = df[~df["URL"].str.contains(pattern, case=False, na=False)].reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"  Step 2 — {removed} login-walled URLs removed (facebook/instagram/tiktok/twitter) → {len(df)} URLs")

    # ── Step 3: Remove dead URLs ──────────────────────────────────────────────
    print(f"  Step 3 — checking {len(df)} URLs for 404s (this may take a minute)...")
    alive = df["URL"].apply(_is_accessible)
    before = len(df)
    df = df[alive].reset_index(drop=True)
    print(f"  Step 3 — {before - len(df)} dead URLs removed → {len(df)} URLs")

    # ── Step 4: Fill missing coordinates ─────────────────────────────────────
    if "coords_approx" not in df.columns:
        df["coords_approx"] = False

    # normalise: treat empty strings as NaN
    df["Lat"] = pd.to_numeric(df["Lat"], errors="coerce")
    df["Lng"] = pd.to_numeric(df["Lng"], errors="coerce")

    missing_mask = df["Lat"].isna()
    n_missing = int(missing_mask.sum())

    if n_missing:
        print(f"  Step 4 — geocoding {n_missing} URLs with missing coordinates...")
        geolocator = Nominatim(user_agent="mt_multisumm_cleaner")
        geocode    = RateLimiter(geolocator.geocode, min_delay_seconds=1)

        for idx in df[missing_mask].index:
            url     = df.at[idx, "URL"]
            address = _scrape_address(url, geo_cfg["postcode_re"], geo_cfg["text_keywords"])
            loc     = geocode(f"{address}, {geo_cfg['geocode_suffix']}") if address else None

            if loc:
                df.at[idx, "Lat"] = loc.latitude
                df.at[idx, "Lng"] = loc.longitude
            else:
                df.at[idx, "Lat"]          = center_lat
                df.at[idx, "Lng"]          = center_lng
                df.at[idx, "coords_approx"] = True
            time.sleep(0.1)

    approx = int(df["coords_approx"].sum())
    print(f"  Step 4 — {approx} URLs assigned approximate city-centre coordinates")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[["URL", "Lat", "Lng", "coords_approx"]].to_csv(out_path, index=False)
    print(f"  ✓ {len(df)} cleaned URLs saved → {out_path}")
    return df
