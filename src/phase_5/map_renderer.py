import io, base64
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
import pandas as pd

_DENSITY_COLORS = {
    "green":  "#27ae60",
    "yellow": "#f39c12",
    "red":    "#e74c3c",
    "none":   "#f0f0f0",
}


def _density_categories(district_counts: Counter) -> dict[str, str]:
    counts = sorted(c for c in district_counts.values() if c > 0)
    if not counts:
        return {}
    n = len(counts)
    hi_thresh = counts[max(0, int(0.75 * n))]
    lo_thresh = counts[max(0, int(0.25 * n))]
    result = {}
    for name, count in district_counts.items():
        if count == 0:
            result[name] = "none"
        elif count >= hi_thresh:
            result[name] = "green"
        elif count > lo_thresh:
            result[name] = "yellow"
        else:
            result[name] = "red"
    return result


def render_map(records: list[dict], cfg: dict) -> str | None:
    districts_path = cfg.get("districts_file")
    if not districts_path or not Path(str(districts_path)).exists():
        return None

    # Load district polygons (already WGS84 after load_districts, but reproject here
    # directly to be self-contained)
    gdf = gpd.read_file(str(districts_path)).to_crs(epsg=4326)

    # Count FSIs per district — exclude nearest-matched (outside boundary) records
    district_counts: Counter = Counter()
    for r in records:
        loc = r.get("location", {})
        name    = loc.get("district_name", "")
        outside = loc.get("match_type") == "nearest"
        if name and name != "Not Specified" and not outside:
            district_counts[name] += 1

    if not district_counts:
        return None

    # Merge counts and density categories into geodataframe
    density_cats = _density_categories(district_counts)
    count_df = pd.DataFrame(
        [{"name": k, "fsi_count": v} for k, v in district_counts.items()]
    )
    if not count_df.empty:
        gdf = gdf.merge(count_df, on="name", how="left")
    gdf["fsi_count"] = gdf["fsi_count"].fillna(0).astype(int)
    gdf["density_cat"] = gdf["name"].map(density_cats).fillna("none")
    gdf["plot_color"]  = gdf["density_cat"].map(_DENSITY_COLORS)

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 10))

    # Choropleth — districts shaded by discrete density category
    gdf.plot(
        ax=ax,
        color=gdf["plot_color"],
        edgecolor="#555",
        linewidth=0.7,
    )

    # Label each district (only those with FSIs, to reduce clutter)
    for _, row in gdf.iterrows():
        if row["fsi_count"] == 0:
            continue
        c = row.geometry.centroid
        ax.annotate(
            f"{row['name']}\n({row['fsi_count']})",
            xy=(c.x, c.y),
            ha="center", va="center",
            fontsize=6, color="#1a1a1a", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.5, ec="none"),
        )

    # Legend
    legend_handles = [
        mpatches.Patch(color=_DENSITY_COLORS["green"],  label="High density"),
        mpatches.Patch(color=_DENSITY_COLORS["yellow"], label="Medium density"),
        mpatches.Patch(color=_DENSITY_COLORS["red"],    label="Low density"),
        mpatches.Patch(color=_DENSITY_COLORS["none"],   label="No FSIs"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=8, framealpha=0.85)

    city = cfg.get("city", "")
    ax.set_title(f"Food Sharing Initiatives — {city}", fontsize=13,
                 fontweight="bold", pad=10)
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)
    ax.tick_params(labelsize=7)

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()
