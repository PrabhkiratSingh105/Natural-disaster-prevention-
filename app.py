import requests
import h5py
import time
import sys


NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

HEADERS = {
    "User-Agent": "DisasterPreventionApp/1.0"
}


def get_city_boundary(city):
    """Find the OpenStreetMap boundary for a city."""

    params = {
        "q": city,
        "format": "json",
        "limit": 1,
        "polygon_geojson": 1
    }

    response = requests.get(
        NOMINATIM_URL,
        params=params,
        headers=HEADERS,
        timeout=30
    )

    response.raise_for_status()

    results = response.json()

    if not results:
        raise Exception(f"Could not find city: {city}")

    result = results[0]

    print(f"Found: {result['display_name']}")
    print(f"OSM ID: {result['osm_id']}")

    return result


def get_areas(city_data):
    """Get local areas inside the city's boundary."""

    osm_type = city_data["osm_type"]
    osm_id = city_data["osm_id"]

    # Nominatim IDs:
    # relation -> relation
    # way      -> way
    # node     -> node

    if osm_type == "relation":
        area_id = 3600000000 + int(osm_id)
    elif osm_type == "way":
        area_id = 2400000000 + int(osm_id)
    elif osm_type == "node":
        area_id = 1600000000 + int(osm_id)
    else:
        raise Exception(f"Unknown OSM type: {osm_type}")

    query = f"""
    [out:json][timeout:120];

    area({area_id})->.searchArea;

    (
        nwr["place"="suburb"](area.searchArea);
        nwr["place"="neighbourhood"](area.searchArea);
        nwr["place"="quarter"](area.searchArea);
        nwr["place"="locality"](area.searchArea);
        nwr["place"="village"](area.searchArea);
        nwr["place"="town"](area.searchArea);
        nwr["place"="city_district"](area.searchArea);
    );

    out center;
    """

    print("Querying OpenStreetMap...")
    
    response = requests.post(
        OVERPASS_URL,
        data=query,
        headers=HEADERS,
        timeout=180
    )

    response.raise_for_status()

    return response.json()["elements"]


def extract_areas(elements):
    """Convert OSM results into clean location records."""

    areas = []
    seen = set()

    for element in elements:

        tags = element.get("tags", {})

        name = tags.get("name")

        if not name:
            continue

        # Nodes have lat/lon directly
        if element["type"] == "node":

            lat = element.get("lat")
            lon = element.get("lon")

        # Ways/relations have a center
        else:

            center = element.get("center", {})

            lat = center.get("lat")
            lon = center.get("lon")

        if lat is None or lon is None:
            continue

        # Remove duplicate names
        key = (name.lower(), round(lat, 5), round(lon, 5))

        if key in seen:
            continue

        seen.add(key)

        areas.append({
            "name": name,
            "latitude": lat,
            "longitude": lon,
            "type": tags.get("place", "unknown")
        })

    areas.sort(key=lambda x: x["name"].lower())

    return areas


# def save_h5(city, areas, filename):
#     """Save locations into an HDF5 file."""

#     names = [x["name"] for x in areas]
#     latitudes = [x["latitude"] for x in areas]
#     longitudes = [x["longitude"] for x in areas]
#     types = [x["type"] for x in areas]

#     with h5py.File(filename, "w") as f:

#         city_group = f.create_group(city)

#         # UTF-8 strings
#         string_dtype = h5py.string_dtype(encoding="utf-8")

#         city_group.create_dataset(
#             "name",
#             data=names,
#             dtype=string_dtype
#         )

#         city_group.create_dataset(
#             "latitude",
#             data=latitudes
#         )

#         city_group.create_dataset(
#             "longitude",
#             data=longitudes
#         )

#         city_group.create_dataset(
#             "type",
#             data=types,
#             dtype=string_dtype
#         )

#     print(f"\nSaved {len(areas)} locations to {filename}")


def main():

    if len(sys.argv) < 2:
        city = input("Enter city name: ").strip()
    else:
        city = " ".join(sys.argv[1:])

    if not city:
        print("Please enter a city.")
        return

    print(f"\nSearching for: {city}\n")

    city_data = get_city_boundary(city)

    time.sleep(1)

    elements = get_areas(city_data)

    areas = extract_areas(elements)

    if not areas:
        print("No areas found.")
        return

    print(f"\nFound {len(areas)} areas:\n")

    for area in areas:
        print(
            f"{area['name']:30} "
            f"{area['latitude']:.6f}, "
            f"{area['longitude']:.6f} "
            f"({area['type']})"
        )

    filename = city.lower().replace(" ", "_") + "_locations.h5"

    # save_h5(city, areas, filename)


if __name__ == "__main__":
    main()