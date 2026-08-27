from pathlib import Path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config_loader import IMAGE


def score_image(img: dict) -> float:
    """
    Scores an image for representativeness.
    Higher = better candidate for the summary report.
    """
    score = 0.0
    w = img.get("width",  0)
    h = img.get("height", 0)

    if w < IMAGE["min_width"] or h < IMAGE["min_height"]:
        return -1.0   # disqualified

    # prefer landscape images
    if IMAGE["prefer_landscape"] and w > h:
        score += 2.0

    # prefer larger images
    if w >= 800:
        score += 2.0
    elif w >= 500:
        score += 1.0

    # prefer images that have already been captioned (Phase 2 placeholder)
    if img.get("caption"):
        score += 1.0

    # prefer images with a real source URL (not data URIs)
    if img.get("source_url", "").startswith("http"):
        score += 0.5

    return score


def select_best_image(images: list[dict]) -> dict | None:
    """Returns the best image from a list, or None if none qualify."""
    if not images:
        return None
    scored = [(score_image(img), img) for img in images]
    scored = [(s, img) for s, img in scored if s >= 0]
    if not scored:
        return None
    return max(scored, key=lambda x: x[0])[1]


def select_images_for_district(initiatives: list[dict],
                                max_images: int = 3) -> list[dict]:
    """
    Selects up to max_images representative images for a district.
    Picks the best image from the most popular initiatives first.
    """
    selected = []
    seen_paths = set()

    for initiative in initiatives:   # already sorted by popularity_score desc
        if len(selected) >= max_images:
            break
        best = select_best_image(initiative.get("images", []))
        if best and best.get("local_path") not in seen_paths:
            selected.append({
                **best,
                "initiative_title": initiative["title"],
                "initiative_url":   initiative["url"],
            })
            seen_paths.add(best.get("local_path"))

    return selected


def select_images_for_city(districts: dict,
                            max_images: int = 5) -> list[dict]:
    """
    Selects top representative images for the city-level overview.
    One per district, from the highest-popularity initiatives.
    """
    selected = []
    seen_paths = set()

    for district_name, d in districts.items():
        if len(selected) >= max_images:
            break
        best = select_best_image(
            [img for initiative in d["initiatives"][:3]
                 for img in initiative.get("images", [])]
        )
        if best and best.get("local_path") not in seen_paths:
            selected.append({
                **best,
                "district": district_name,
            })
            seen_paths.add(best.get("local_path"))

    return selected