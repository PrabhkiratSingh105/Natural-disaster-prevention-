import os

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
HEADERS = {
    "User-Agent": "NaturalDisasterPrevention/1.0 (boundary lookup)"
}


def flatten_coordinates(geometry):
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])

    if geometry_type == "GeometryCollection":
        return [
            point
            for child in geometry.get("geometries", [])
            for point in flatten_coordinates(child)
        ]
    if geometry_type == "Point":
        return [coordinates]
    if geometry_type in {"LineString", "MultiPoint"}:
        return coordinates
    if geometry_type in {"Polygon", "MultiLineString"}:
        return [point for ring in coordinates for point in ring]
    if geometry_type == "MultiPolygon":
        return [
            point
            for polygon in coordinates
            for ring in polygon
            for point in ring
        ]
    return []


def boundary_extremes(location):
    response = requests.get(
        NOMINATIM_URL,
        params={
            "q": location,
            "format": "jsonv2",
            "polygon_geojson": 1,
            "limit": 1,
        },
        headers=HEADERS,
        timeout=30,
    )
    response.raise_for_status()
    results = response.json()
    if not results:
        raise ValueError(f"No boundary found for {location}.")

    result = results[0]
    coordinates = [
        point
        for point in flatten_coordinates(result.get("geojson", {}))
        if len(point) >= 2
    ]
    if not coordinates:
        raise ValueError(
            f"No polygon coordinates were returned for {location}; viewport data is not enough."
        )

    north = max(coordinates, key=lambda point: point[1])
    south = min(coordinates, key=lambda point: point[1])
    east = max(coordinates, key=lambda point: point[0])
    west = min(coordinates, key=lambda point: point[0])

    def point_data(point):
        return {"lat": round(float(point[1]), 6), "lon": round(float(point[0]), 6)}

    return {
        "location": result.get("display_name", location),
        "extreme_points": {
            "north": point_data(north),
            "south": point_data(south),
            "east": point_data(east),
            "west": point_data(west),
        },
        "limits": {
            "north_lat": round(float(north[1]), 6),
            "south_lat": round(float(south[1]), 6),
            "east_lon": round(float(east[0]), 6),
            "west_lon": round(float(west[0]), 6),
        },
        "geojson": result.get("geojson"),
    }


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/api/bounds")
def bounds():
    payload = request.get_json(silent=True) or {}
    location = str(payload.get("location", "")).strip()
    if not location:
        return jsonify({"error": "Provide a selected location."}), 400

    try:
        return jsonify(boundary_extremes(location))
    except requests.RequestException as error:
        return jsonify({"error": f"Boundary service request failed: {error}"}), 502
    except ValueError as error:
        return jsonify({"error": str(error)}), 404


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        debug=True,
    )
