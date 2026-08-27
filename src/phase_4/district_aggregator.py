from collections import Counter, defaultdict


def aggregate_by_district(records: list[dict]) -> dict:
    """
    Groups FSI records by district and computes per-district statistics.
    Returns a dict keyed by district_name.
    """
    districts = defaultdict(lambda: {
        "district_name":  "",
        "district_code":  "",
        "fsi_count":      0,
        "fsi_types":      Counter(),
        "operational_levels": Counter(),
        "popularity_levels": Counter(),
        "initiatives":    [],
    })

    for r in records:
        loc  = r.get("location", {})
        name = loc.get("district_name", "Unknown")
        code = loc.get("district_code", "Unknown")
        d    = districts[name]

        d["district_name"]  = name
        d["district_code"]  = code
        d["fsi_count"]     += 1
        d["fsi_types"][r.get("fsi_type", "unknown")]             += 1
        d["operational_levels"][r.get("operational_level", "unknown")] += 1
        d["popularity_levels"][r.get("popularity", {})
                                 .get("popularity_level", "unknown")] += 1
        d["initiatives"].append({
            "title":             r.get("title", ""),
            "url":               r.get("url", ""),
            "fsi_type":          r.get("fsi_type", "unknown"),
            "operational_level": r.get("operational_level", "unknown"),
            "popularity_level":  r.get("popularity", {}).get("popularity_level", "unknown"),
            "popularity_score":  r.get("popularity", {}).get("popularity_score", 0),
            "description":       r.get("description", ""),
            "images":            r.get("images", []),
            "lat":               loc.get("lat"),
            "lng":               loc.get("lng"),
        })

    # convert Counters to plain dicts and sort districts by FSI count
    result = {}
    for name, d in sorted(districts.items(),
                           key=lambda x: x[1]["fsi_count"], reverse=True):
        result[name] = {
            **d,
            "fsi_types":          dict(d["fsi_types"].most_common()),
            "operational_levels": dict(d["operational_levels"].most_common()),
            "popularity_levels":  dict(d["popularity_levels"].most_common()),
            # sort initiatives within district by popularity score descending
            "initiatives": sorted(d["initiatives"],
                                  key=lambda x: x["popularity_score"],
                                  reverse=True),
        }

    _assign_density_categories(result)
    return result


def _assign_density_categories(districts: dict) -> None:
    """Assign green/yellow/red density label to each district in-place."""
    counts = sorted(d["fsi_count"] for d in districts.values() if d["fsi_count"] > 0)
    if not counts:
        for d in districts.values():
            d["density_category"] = "none"
        return
    n = len(counts)
    hi_thresh = counts[max(0, int(0.75 * n))]
    lo_thresh = counts[max(0, int(0.25 * n))]
    for d in districts.values():
        c = d["fsi_count"]
        if c == 0:
            d["density_category"] = "none"
        elif c >= hi_thresh:
            d["density_category"] = "green"
        elif c > lo_thresh:
            d["density_category"] = "yellow"
        else:
            d["density_category"] = "red"


def compute_city_stats(records: list[dict]) -> dict:
    """Top-level city-wide statistics."""
    return {
        "total_fsi_count":    len(records),
        "district_count":     len(set(
            r.get("location", {}).get("district_name", "Unknown")
            for r in records
        )),
        "fsi_type_breakdown": dict(
            Counter(r.get("fsi_type", "unknown") for r in records).most_common()
        ),
        "operational_breakdown": dict(
            Counter(r.get("operational_level", "unknown")
                    for r in records).most_common()
        ),
        "popularity_breakdown": dict(
            Counter(r.get("popularity", {}).get("popularity_level", "unknown")
                    for r in records).most_common()
        ),
    }