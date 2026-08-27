from pathlib import Path


def score_image(img: dict) -> float:
    w = img.get("width",  0)
    h = img.get("height", 0)
    if w < 200 or h < 150:
        return -1.0
    score = 0.0
    if w > h:         score += 2.0   # landscape
    if w >= 800:      score += 2.0
    elif w >= 500:    score += 1.0
    if img.get("source_url", "").startswith("http"):
        score += 0.5
    return score


def select_representative_images(records: list[dict],
                                  max_images: int = 4) -> list[dict]:
    """
    Selects the best images across all FSIs, ranked by image quality
    and FSI popularity score. Returns images with FSI title as caption.
    """
    # sort records by popularity score descending
    sorted_records = sorted(
        records,
        key=lambda r: r.get("popularity", {}).get("popularity_score", 0),
        reverse=True,
    )

    selected   = []
    seen_paths = set()

    for record in sorted_records:
        if len(selected) >= max_images:
            break
        images = record.get("images", [])
        if not images:
            continue

        # pick best image from this FSI
        scored = [(score_image(img), img) for img in images]
        scored = [(s, img) for s, img in scored if s >= 0]
        if not scored:
            continue

        best_score, best_img = max(scored, key=lambda x: x[0])
        path = best_img.get("local_path", "")

        if path in seen_paths:
            continue
        seen_paths.add(path)

        selected.append({
            **best_img,
            "caption": record.get("title", "Unknown FSI"),
            "source_page": record.get("url", ""),
        })

    return selected