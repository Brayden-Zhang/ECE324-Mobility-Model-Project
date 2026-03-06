import argparse
import json

import numpy as np

try:
    import osmnx as ox
except Exception as exc:
    raise RuntimeError("osmnx is required for OSM context building") from exc

try:
    import h3
except Exception as exc:
    raise RuntimeError("h3 is required for OSM context building") from exc


def parse_args():
    parser = argparse.ArgumentParser(description="Build H3-cell OSM context features")
    parser.add_argument("--north", type=float, required=True)
    parser.add_argument("--south", type=float, required=True)
    parser.add_argument("--east", type=float, required=True)
    parser.add_argument("--west", type=float, required=True)
    parser.add_argument("--res", type=int, default=7)
    parser.add_argument("--output", type=str, default="data/osm_context.json")
    return parser.parse_args()


def main():
    args = parse_args()
    tags = {
        "amenity": True,
        "shop": True,
        "leisure": True,
        "highway": True,
    }
    gdf = ox.geometries_from_bbox(args.north, args.south, args.east, args.west, tags)
    context = {}

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        if geom.geom_type in (
            "Point",
            "Polygon",
            "MultiPolygon",
            "LineString",
            "MultiLineString",
        ):
            if geom.geom_type == "Point":
                lat, lon = geom.y, geom.x
            else:
                centroid = geom.centroid
                lat, lon = centroid.y, centroid.x
            cell = (
                h3.latlng_to_cell(lat, lon, args.res)
                if hasattr(h3, "latlng_to_cell")
                else h3.geo_to_h3(lat, lon, args.res)
            )
            key = str(cell)
            if key not in context:
                context[key] = np.zeros((16,), dtype=np.float32)
            # simple feature buckets: amenity/shop/leisure/highway presence counts
            if row.get("amenity") is not None:
                context[key][0] += 1
            if row.get("shop") is not None:
                context[key][1] += 1
            if row.get("leisure") is not None:
                context[key][2] += 1
            if row.get("highway") is not None:
                context[key][3] += 1

    serialized = {k: v.tolist() for k, v in context.items()}
    with open(args.output, "w") as f:
        json.dump(serialized, f)
    print(f"saved {len(serialized)} cells to {args.output}")


if __name__ == "__main__":
    main()
