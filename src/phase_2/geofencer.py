import geopandas as gpd
from shapely.geometry import Point


def load_districts(geojson_path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(geojson_path)
    gdf = gdf.to_crs(epsg=4326)
    return gdf


def assign_district(lat: float, lng: float,
                    districts: gpd.GeoDataFrame) -> dict:
    point = Point(lng, lat)

    matches = districts[districts.geometry.contains(point)]
    if not matches.empty:
        if len(matches) == 1:
            row = matches.iloc[0]
        else:
            # multiple polygons overlap — pick the nearest centroid
            matches_proj = matches.to_crs(epsg=3857)
            point_proj   = gpd.GeoSeries([point], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]
            distances    = matches_proj.geometry.centroid.distance(point_proj)
            row          = matches.iloc[distances.argmin()]
        return {
            "district_code": row.get("district_code", row.get("name", "Unknown")),
            "district_name": row.get("name", "Unknown"),
            "match_type":    "exact",
        }

    # point outside all polygons — assign nearest centroid
    districts_proj = districts.to_crs(epsg=3857)
    point_proj     = gpd.GeoSeries([point], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]
    distances      = districts_proj.geometry.centroid.distance(point_proj)
    nearest        = districts.iloc[distances.argmin()]

    return {
        "district_code": nearest.get("district_code", nearest.get("name", "Unknown")),
        "district_name": nearest.get("name", "Unknown"),
        "match_type":    "nearest",
    }


def geofence_all(records: list[dict],
                 districts: gpd.GeoDataFrame,
                 **_kwargs) -> list[dict]:
    exact = 0
    nearest = 0
    no_coords = 0

    for r in records:
        geo = r.get("geo")
        if not geo or not geo.get("lat") or not geo.get("lng"):
            r["district"] = {
                "district_code": "Not Specified",
                "district_name": "Not Specified",
                "match_type":    "no_coords",
            }
            no_coords += 1
            continue

        result = assign_district(geo["lat"], geo["lng"], districts)
        r["district"] = result
        if result["match_type"] == "exact":
            exact += 1
        else:
            nearest += 1

    print(f"  District assignment: {exact} exact, {nearest} nearest-centroid, "
          f"{no_coords} no coords")
    return records
