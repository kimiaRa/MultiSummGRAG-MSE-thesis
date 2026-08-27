import json
from pathlib import Path
from district_aggregator import aggregate_by_district, compute_city_stats
from image_selector import select_images_for_district, select_images_for_city
from text_analytics import analyse_records


def build_schema(records: list[dict],
                 graph_answers: dict,
                 city_cfg: dict) -> dict:
    districts  = aggregate_by_district(records)
    city_stats = compute_city_stats(records)
    analytics  = analyse_records(records)        # new

    for name, d in districts.items():
        d["representative_images"] = select_images_for_district(
            d["initiatives"], max_images=3
        )

    city_images = select_images_for_city(districts, max_images=5)

    return {
        "meta": {
            "country": city_cfg["country"],
            "city":    city_cfg["city"],
            "version": 5,
        },
        "city_overview": {
            "stats":                  city_stats,
            "analytics":              analytics,          # new
            "representative_images":  city_images,
            "geographic_distribution": graph_answers.get(
                "geographic_distribution", {}).get("answer", ""),
            "fsi_types_summary":      graph_answers.get(
                "fsi_types", {}).get("answer", ""),
            "operational_summary":    graph_answers.get(
                "operational_levels", {}).get("answer", ""),
            "popularity_summary":     graph_answers.get(
                "popularity", {}).get("answer", ""),
            "notable_initiatives":    graph_answers.get(
                "notable_initiatives", {}).get("answer", ""),
            "hotspots_and_deserts":   graph_answers.get(
                "hotspots_and_deserts", {}).get("answer", ""),
            "digital_presence":       graph_answers.get(
                "digital_presence", {}).get("answer", ""),
            "network_affiliations":   graph_answers.get(
                "network_affiliations", {}).get("answer", ""),
            "activity_volume":        graph_answers.get(
                "activity_volume", {}).get("answer", ""),
            "operating_hours":        graph_answers.get(
                "operating_hours", {}).get("answer", ""),
            "physical_vs_online":     graph_answers.get(
                "physical_vs_online", {}).get("answer", ""),
            "target_audiences":       graph_answers.get(
                "target_audiences", {}).get("answer", ""),
            "sustainability":         graph_answers.get(
                "sustainability", {}).get("answer", ""),
            "collaborations":         graph_answers.get(
                "collaborations", {}).get("answer", ""),
            "education_and_engagement": graph_answers.get(
                "education_and_engagement", {}).get("answer", ""),
            "events_and_workshops":   graph_answers.get(
                "events_and_workshops", {}).get("answer", ""),
        },
        "districts": districts,
    }