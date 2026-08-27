from urllib.parse import urlparse


def _normalise(url: str) -> str:
    """Strip trailing slash, www, and fragment for comparison."""
    parsed = urlparse(url.strip().lower())
    host   = parsed.netloc.replace("www.", "")
    path   = parsed.path.rstrip("/")
    return f"{host}{path}"


def detect_duplicates(rows: list[dict]) -> list[dict]:
    """
    Flags exact and near-duplicate URLs.
    Near-duplicates share the same normalised URL.
    """
    seen      = {}   # normalised_url -> first row index
    for i, row in enumerate(rows):
        norm = _normalise(row["URL"])
        if norm in seen:
            row["is_duplicate"] = True
            row["duplicate_of"] = rows[seen[norm]]["URL"]
        else:
            row["is_duplicate"] = False
            row["duplicate_of"] = None
            seen[norm] = i
    return rows