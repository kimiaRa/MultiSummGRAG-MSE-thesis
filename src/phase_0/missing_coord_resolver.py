def resolve_missing_coords(rows: list[dict],
                            city: str,
                            country: str) -> list[dict]:
    """
    Flags rows with missing coordinates for manual review.
    Only touches rows where Lat or Lng is genuinely empty.
    """
    missing = [
        r for r in rows
        if not r.get("Lat") or str(r.get("Lat", "")).strip() == ""
        or not r.get("Lng") or str(r.get("Lng", "")).strip() == ""
    ]

    print(f"  Flagging {len(missing)} rows with missing coordinates...")

    for row in missing:
        row["Lat"]            = ""
        row["Lng"]            = ""
        row["coord_source"]   = "not_specified"
        row["coord_strategy"] = None
        row["needs_review"]   = True

    # mark rows that already have coordinates
    for row in rows:
        if row not in missing:
            row["coord_source"]   = "provided"
            row["coord_strategy"] = None
            row["needs_review"]   = False

    return rows