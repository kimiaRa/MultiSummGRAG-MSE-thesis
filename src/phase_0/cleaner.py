import csv
from pathlib import Path


# def write_clean_csv(rows: list[dict], out_path: Path) -> list[dict]:
#     """
#     Writes ALL non-duplicate rows to urls.csv regardless of FSI detection.
#     Phase 0 is analysis-only — no data is removed.
#     Flags are preserved in the annotated JSON for thesis reporting.
#     """
#     # only skip true duplicates — same URL appearing twice
#     clean = [r for r in rows if not r.get("is_duplicate")]

#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     with open(out_path, "w", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=["URL", "Lat", "Lng"])
#         writer.writeheader()
#         for r in clean:
#             writer.writerow({
#                 "URL": r["URL"],
#                 "Lat": r.get("Lat", ""),
#                 "Lng": r.get("Lng", ""),
#             })

#     with_coords = sum(
#         1 for r in clean
#         if r.get("Lat") and str(r.get("Lat", "")).strip() != ""
#     )
#     non_fsi = sum(1 for r in clean if r.get("is_non_fsi"))
#     print(f"  ✓ Clean CSV written → {out_path} "
#           f"({len(clean)} URLs, {with_coords} with coordinates)")
#     print(f"  ℹ {non_fsi} flagged as likely non-FSI "
#           f"(kept for completeness, documented in phase0_review.json)")
#     return clean


def load_raw_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # strip whitespace from column names
        fieldnames = [f.strip() for f in reader.fieldnames]

        for row in reader:
            # normalise keys — strip whitespace
            row = {k.strip(): v.strip() for k, v in row.items()}

            if not row.get("URL", "").strip():
                continue

            # format 1: separate Lat and Lng columns (Dublin style)
            if "Lat" in fieldnames and "Lng" in fieldnames:
                lat = row.get("Lat", "").strip()
                lng = row.get("Lng", "").strip()

            # format 2: single coordinates column "lat,lng" (Cork style)
            elif "coordinates" in fieldnames or "Coordinates" in fieldnames:
                coords_raw = row.get("coordinates") or row.get("Coordinates", "")
                if coords_raw and "," in coords_raw:
                    parts = coords_raw.split(",")
                    lat   = parts[0].strip()
                    lng   = parts[1].strip()
                else:
                    lat = ""
                    lng = ""

            # format 3: fallback — no coords at all
            else:
                lat = ""
                lng = ""

            rows.append({
                "URL": row["URL"].strip(),
                "Lat": lat,
                "Lng": lng,
            })

    return rows

