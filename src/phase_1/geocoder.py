import re
import time
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

_geo = Nominatim(user_agent="fsi-thesis-phase1")

# address patterns to extract from page text
_COUNTRY_CODES = {
    "Ireland":        ["ie"],
    "United Kingdom": ["gb"],
    "Spain":          ["es"],
    "Italy":          ["it"],
    "France":         ["fr"],
    "Germany":        ["de"],
    "Netherlands":    ["nl"],
}

_ADDRESS_PATTERNS = [
    # UK postcode
    r'\b([A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2})\b',
    # Irish Eircode
    r'\b([A-Z]\d{2}\s*[A-Z0-9]{4})\b',
    # Spanish postal code (5 digits, starting 08 for Barcelona)
    r'\b(0[0-9]\d{3})\b',
    # Italian CAP (5 digits, starting 20 for Milan)
    r'\b(20\d{3})\b',
    # English street address
    r'\d{1,4}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*'
    r'(?:\s+(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr'
    r'|Place|Pl|Close|Row|Square|Sq|Way|Gardens|Terrace))',
    # Spanish street address (Carrer, Carretera, Avinguda, Passeig, Plaça)
    r'(?:Carrer|Carretera|Avinguda|Passeig|Pla[çc]a)\s+[A-Za-zÀ-ú]+(?:\s+[A-Za-zÀ-ú]+)*,?\s*\d{1,4}',
    # Italian street address (Via, Corso, Piazza, Viale, Largo)
    r'(?:Via|Corso|Piazza|Viale|Largo)\s+[A-Za-zÀ-ú]+(?:\s+[A-Za-zÀ-ú]+)*,?\s*\d{1,4}',
]

# phrases that signal a location in text
_LOCATION_PATTERNS = [
    r'based in ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
    r'located in ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
    r'serving ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
    r'in the ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?) area',
    # Spanish location phrases
    r'ubicad[ao] en ([A-Za-zÀ-ú]+(?:\s+[A-Za-zÀ-ú]+)?)',
    r'situat[ao] en ([A-Za-zÀ-ú]+(?:\s+[A-Za-zÀ-ú]+)?)',
    # Italian location phrases
    r'situato a ([A-Za-zÀ-ú]+(?:\s+[A-Za-zÀ-ú]+)?)',
    r'si trova a ([A-Za-zÀ-ú]+(?:\s+[A-Za-zÀ-ú]+)?)',
]


def _extract_candidates(text: str) -> list[str]:
    candidates = []
    for pat in _ADDRESS_PATTERNS:
        candidates.extend(re.findall(pat, text))
    for pat in _LOCATION_PATTERNS:
        candidates.extend(re.findall(pat, text, re.IGNORECASE))
    return list(dict.fromkeys(candidates))[:5]


def _geocode_query(query: str, city: str,
                   country_codes: list[str],
                   retries: int = 3) -> dict | None:
    for attempt in range(retries):
        try:
            loc = _geo.geocode(
                f"{query}, {city}",
                country_codes=country_codes,
                timeout=10,
            )
            if loc:
                return {
                    "lat":    loc.latitude,
                    "lng":    loc.longitude,
                    "source": "text_extracted",
                    "query":  query,
                }
            return None
        except GeocoderTimedOut:
            time.sleep(1.5 ** attempt)
        except GeocoderServiceError:
            time.sleep(2)
    return None


def geocode_fsi(scraped: dict, city: str = "",
                country: str = "") -> dict | None:
    """
    Returns coordinates for an FSI.
    Priority:
      1. Provided coordinates from CSV (most accurate)
      2. Text-extracted address from scraped page
      3. None (excluded from geofencing)
    """
    # priority 1 — provided coordinates
    try:
        lat = scraped.get("Lat")
        lng = scraped.get("Lng")
        if lat and lng and str(lat).strip() and str(lng).strip():
            return {
                "lat":          float(lat),
                "lng":          float(lng),
                "source":       "provided",
                "coords_approx": scraped.get("coords_approx", False),
            }
    except (ValueError, TypeError):
        pass

    # priority 2 — extract from scraped text
    text = scraped.get("html_text", "") or scraped.get("text", "") or ""
    if text and city:
        country_codes = _COUNTRY_CODES.get(country, ["gb", "ie"])
        candidates    = _extract_candidates(text)

        for candidate in candidates:
            result = _geocode_query(candidate, city, country_codes)
            if result:
                time.sleep(1.1)   # Nominatim rate limit
                return result
            time.sleep(1.1)

    # priority 3 — no coordinates available
    return None