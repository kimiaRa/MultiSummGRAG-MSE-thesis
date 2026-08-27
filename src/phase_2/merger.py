from pathlib import Path


def build_enriched_record(record: dict, city_cfg: dict) -> dict:
    district = record.get("district") or {}
    geo      = record.get("geo") or {}        # fix: handles None explicitly

    return {
        "url":   record["url"],
        "title": record["title"],
        "text":  record["text"],

        "location": {
            "country":       city_cfg["country"],
            "city":          city_cfg["city"],
            "district_name": district.get("district_name", "Unknown"),
            "district_code": district.get("district_code", "Unknown"),
            "match_type":    district.get("match_type", "unknown"),
            "lat":           geo.get("lat"),
            "lng":           geo.get("lng"),
            "coords_approx": geo.get("coords_approx", False),
        },

        "fsi_type":          record.get("fsi_type", "unknown"),
        "operational_level": record.get("operational_level", "unknown"),
        "popularity":        record.get("popularity", {}),
        "images":            record.get("images", []),
        "description":       record.get("description", ""),
        "scraped_at":        record.get("scraped_at"),
        "error":             record.get("error"),
        "phase":             2,
    }


def merge_all(records: list[dict], city_cfg: dict) -> list[dict]:
    return [build_enriched_record(r, city_cfg) for r in records]