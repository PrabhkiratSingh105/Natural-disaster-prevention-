from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr
import rioxarray  # noqa: F401  (registers the .rio accessor on xarray objects)
import rasterio
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import folium
from folium.plugins import HeatMap


# ==========================================================================
# CONFIG -- edit these
# ==========================================================================

ELEVATION_NC_PATH = "data/gebco_2026_sub_ice_n34.954_s2.877_w63.003_e90.804.nc"  # your downloaded GEBCO file
POPULATION_TIF_PATH = "data/ind_pop_2026_CN_100m_R2025A_v1.tif"    # your downloaded WorldPop file

# India bounding box (padded slightly beyond the coastline so offshore wind
# field values aren't clipped before landfall)   # ~11km cells; raise to 0.25 for faster/coarser

# Storm parameters -- replace with your model's actual inference output


# Composite damage index weights -- tune these, they don't need to sum to 1
WEIGHTS = {
    "wind": 0.5,
    "population": 0.3,
    "low_elevation": 0.2,
}

# Population/elevation vulnerability has NO distance dependence on its own
# (a city is just as populated whether the storm is near or far) - this
# scales the WHOLE composite index down with distance from the storm
# center, so a location far away doesn't show elevated risk just because
# it happens to be low-lying/populated. ~300km ~= a typical r34 gale-force
# radius; raise for a broader "at risk" footprint, lower for a tighter one.
DISTANCE_DECAY_KM = 300.0

OCEAN_COLOR = "#3a7ca8"  # sea cells are masked out of the risk colormap and drawn this color instead

OUTPUT_PNG = "damage_map.png"
OUTPUT_HTML = "damage_map.html"

# Where the Google Maps frontend bundle (overlay.png + overlay.json) gets
# written. Put index.html in this same folder and serve it locally.
WEB_OUTPUT_DIR = "web"


# ==========================================================================
# Step 1: Build the target grid
# ==========================================================================


def build_grid(lat_min, lat_max, lon_min, lon_max, resolution):
    lats = np.arange(lat_min, lat_max, resolution)
    lons = np.arange(lon_min, lon_max, resolution)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    return lats, lons, lat_grid, lon_grid


# ==========================================================================
# Step 2: Haversine distance (vectorized) from storm center to every cell
# ==========================================================================


def haversine_distance_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# ==========================================================================
# Step 3: Holland wind profile
# ==========================================================================


def fit_holland_B(vmax_kt, mslp_hpa, env_pressure=1013.0):
    """Classic Holland (1980) B parameter, fit from Vmax and central
    pressure deficit:

        B = (rho_air * e * Vmax^2) / (Pn - Pc)

    where Pn is the ambient/environmental pressure (hPa), Pc is the
    storm's central pressure (mslp, hPa), rho_air ~= 1.15 kg/m^3 (typical
    tropical marine boundary layer density), e = Euler's number.

    Physically, B is typically 1.0-2.5; deficient/noisy central-pressure
    estimates (e.g. very weak storms where Pn - Pc is tiny) can push the
    fit outside that range, so we clip to keep the wind profile sane.
    """
    vmax_ms = vmax_kt * 0.514444  # knots -> m/s
    pressure_deficit_pa = max(env_pressure - mslp_hpa, 1.0) * 100.0  # hPa -> Pa, floor to avoid /0
    rho_air = 1.15
    B = (rho_air * np.e * vmax_ms ** 2) / pressure_deficit_pa
    return float(np.clip(B, 1.0, 2.5))


def holland_wind_field(distance_km, vmax_kt, rmw_nm, B=1.5):
    """Returns wind speed (in knots) at each grid cell.

    distance_km : array, distance from storm center to each grid cell (km)
    vmax_kt     : storm's max sustained wind speed (knots)
    rmw_nm      : radius of maximum wind (nautical miles) -- converted to km
    """
    rmw_km = rmw_nm * 1.852  # nautical miles -> km
    r = np.maximum(distance_km, 1e-3)  # avoid divide-by-zero at the exact center
    ratio = (rmw_km / r) ** B
    wind_kt = vmax_kt * np.sqrt(ratio * np.exp(1 - ratio))
    return wind_kt


def build_calibrated_wind_field(distance_km, storm, B=1.5):
    """Wind field built from ALL the radii the model actually outputs
    (rmw, r34avg, r50avg, r64avg), not just vmax+rmw.

    Inside the calibrated range (0 to the outermost known radius), wind
    speed is monotone-interpolated between the real anchor points:
        (0, vmax) -> (rmw, vmax) -> (r64_km, 64kt) -> (r50_km, 50kt) -> (r34_km, 34kt)
    which directly honors mslp-independent structure the model predicted,
    rather than assuming one idealized Holland curve holds everywhere.

    Beyond the outermost known radius (normally r34avg, since 34kt is the
    widest/outermost threshold), there's no more calibration data, so we
    fall back to a Holland-shaped decay anchored at that last known point
    - B here should come from fit_holland_B(vmax, mslp) so the *outer*
    decay is still informed by the storm's pressure-based intensity.

    Any of r34avg/r50avg/r64avg can be missing/NaN (e.g. weak storms
    often have no r64avg) - those anchors are just skipped.
    """
    rmw_km = storm["rmw"] * 1.852
    anchors_r = [0.0, rmw_km]
    anchors_w = [storm["vmax"], storm["vmax"]]

    # Radii shrink as wind threshold rises (r64 < r50 < r34), so walk them
    # from innermost to outermost to build a properly sorted anchor list.
    for key, wind_kt in (("r64avg", 64.0), ("r50avg", 50.0), ("r34avg", 34.0)):
        r_nm = storm.get(key)
        if r_nm is None or (isinstance(r_nm, float) and np.isnan(r_nm)) or r_nm <= 0:
            continue
        r_km = r_nm * 1.852
        if r_km > anchors_r[-1]:  # must be strictly increasing to interpolate safely
            anchors_r.append(r_km)
            anchors_w.append(wind_kt)

    anchors_r = np.asarray(anchors_r, dtype=np.float64)
    anchors_w = np.asarray(anchors_w, dtype=np.float64)

    flat_r = distance_km.ravel()
    # np.interp clips to the boundary value past the range by default; we
    # want to detect "past the range" instead, so mask it explicitly.
    wind = np.interp(flat_r, anchors_r, anchors_w)
    beyond = flat_r > anchors_r[-1]

    if beyond.any():
        r_last, w_last = anchors_r[-1], anchors_w[-1]
        r_beyond = np.maximum(flat_r[beyond], r_last)
        ratio = (r_last / r_beyond) ** B
        # Holland shape, rescaled so it's continuous with w_last at r_last
        # instead of jumping back up to vmax at the (already-passed) rmw.
        holland_at_r_last = np.sqrt((r_last / r_last) ** B * np.exp(1 - (r_last / r_last) ** B))
        holland_at_r = np.sqrt(ratio * np.exp(1 - ratio))
        wind[beyond] = w_last * (holland_at_r / max(holland_at_r_last, 1e-6))

    return wind.reshape(distance_km.shape)


# ==========================================================================
# Step 4: Load and resample elevation (.nc) onto the target grid
# ==========================================================================


def load_elevation_on_grid(nc_path, target_lats, target_lons):
    """Loads a GEBCO-style NetCDF elevation/bathymetry file and resamples it
    onto the target lat/lon grid via linear interpolation.

    NOTE: GEBCO's variable name is typically 'elevation' -- if this raises a
    KeyError, open the file once separately and check `ds.data_vars` to
    confirm the actual variable name in your specific download.
    """
    ds = xr.open_dataset(nc_path)

    var_name = "elevation" if "elevation" in ds.data_vars else list(ds.data_vars)[0]
    elev = ds[var_name]

    # Confirm coordinate names -- GEBCO commonly uses 'lat'/'lon'
    lat_name = "lat" if "lat" in elev.coords else "latitude"
    lon_name = "lon" if "lon" in elev.coords else "longitude"

    elev_on_grid = elev.interp(
        {lat_name: target_lats, lon_name: target_lons}, method="linear"
    )
    return elev_on_grid.values  # shape (n_lats, n_lons)


# ==========================================================================
# Step 5: Load and resample population density (.tif) onto the target grid
# ==========================================================================


def load_population_on_grid(tif_path, target_lats, target_lons):
    """Loads a WorldPop GeoTIFF and resamples it onto the target lat/lon
    grid using AREA-AVERAGE resampling (not point sampling).

    Population is extremely spiky at 100m resolution (a village next to
    empty land) - naive point interpolation onto a much coarser grid
    just picks up whichever single source pixel happens to land nearest
    each grid point, producing salt-and-pepper noise. Averaging every
    source pixel that falls inside each output cell fixes that.
    """
    from rasterio.warp import reproject
    from rasterio.transform import from_bounds

    pop = rioxarray.open_rasterio(tif_path, masked=True).squeeze()
    src_data = pop.values.astype("float32")
    src_transform = pop.rio.transform()
    src_crs = pop.rio.crs

    height, width = len(target_lats), len(target_lons)
    lat_min, lat_max = float(np.min(target_lats)), float(np.max(target_lats))
    lon_min, lon_max = float(np.min(target_lons)), float(np.max(target_lons))
    res_lat = (lat_max - lat_min) / max(height - 1, 1)
    res_lon = (lon_max - lon_min) / max(width - 1, 1)
    # from_bounds wants the FULL extent (cell edges), target_lats/lons are
    # cell CENTERS, so pad out by half a cell on each side.
    dst_transform = from_bounds(
        lon_min - res_lon / 2, lat_min - res_lat / 2,
        lon_max + res_lon / 2, lat_max + res_lat / 2,
        width, height,
    )

    dst = np.full((height, width), np.nan, dtype=np.float32)
    reproject(
        source=src_data,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=np.nan,
        dst_transform=dst_transform,
        dst_crs="EPSG:4326",  # target grid is plain lat/lon
        dst_nodata=np.nan,
        resampling=Resampling.average,  # <- the fix: area-average, not nearest/point
    )

    # from_bounds produces a north-up raster (row 0 = max lat), but our
    # target grid's row 0 = min lat (see build_grid) - flip to match.
    values = np.flipud(dst)

    # WorldPop uses a large negative sentinel (e.g. -99999) for nodata in
    # some products -- clean that up before using it in the composite index
    values = np.where(values < 0, 0, values)
    values = np.nan_to_num(values, nan=0.0)
    return values


# ==========================================================================
# Step 6: Normalize a grid to 0-1 for safe combination into the composite index
# ==========================================================================


def normalize(grid):
    grid = np.nan_to_num(grid, nan=0.0)
    g_min, g_max = np.nanmin(grid), np.nanmax(grid)
    if g_max - g_min < 1e-9:
        return np.zeros_like(grid)
    return (grid - g_min) / (g_max - g_min)


# ==========================================================================
# Step 7: Composite damage index
# ==========================================================================


def compute_damage_index(wind_grid, population_grid, elevation_grid, weights,
                          land_mask, distance_km, decay_km=300.0):
    """land_mask: True where land, False where ocean (elevation_grid >= 0).

    Ocean cells get damage=NaN (masked out of the map entirely, drawn as
    plain ocean color) instead of being folded into the risk colormap -
    a composite "damage to infrastructure/population" index isn't
    meaningful over open water.

    The whole index is also scaled by exp(-distance/decay_km), since
    population and elevation have no distance dependence on their own
    (a city is just as populated regardless of where the storm is) - this
    keeps risk localized around the storm instead of population/elevation
    alone lighting up distant, unaffected regions.
    """
    wind_norm = normalize(wind_grid)
    population_norm = normalize(np.where(land_mask, population_grid, 0.0))

    # Elevation risk only makes sense on land - low-lying LAND is more
    # flood-prone; ocean depth isn't "low elevation risk," it's just sea.
    elevation_clipped = np.clip(elevation_grid, 0, None)
    inverted_elevation = 1.0 / (1.0 + elevation_clipped)
    elevation_norm = normalize(np.where(land_mask, inverted_elevation, 0.0))

    damage = (
        weights["wind"] * wind_norm
        + weights["population"] * population_norm
        + weights["low_elevation"] * elevation_norm
    )
    damage = normalize(damage)

    distance_decay = np.exp(-distance_km / decay_km)
    damage = damage * distance_decay

    damage = np.where(land_mask, damage, np.nan)  # NaN over ocean -> masked in plotting
    return damage


# ==========================================================================
# Step 8: Visualization
# ==========================================================================


def plot_static_map(damage_grid, lat_min, lat_max, lon_min, lon_max, storm_lat, storm_lon, out_path):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor(OCEAN_COLOR)  # shows through wherever damage_grid is NaN (ocean)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "risk", ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e0000"]
    )
    cmap.set_bad(color=OCEAN_COLOR)  # belt-and-suspenders in case imshow doesn't fully composite over facecolor
    masked = np.ma.masked_invalid(damage_grid)
    im = ax.imshow(
        masked,
        cmap=cmap,
        extent=[lon_min, lon_max, lat_min, lat_max],
        origin="lower",
        vmin=0, vmax=1,
    )
    ax.plot(storm_lon, storm_lat, marker="*", color="black", markersize=18, label="Storm center")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Cyclone Damage Risk Index")
    ax.legend(loc="upper right")
    plt.colorbar(im, ax=ax, label="Damage Risk (0-1)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved static map to {out_path}")


def plot_interactive_map(damage_grid, lats, lons, land_mask, storm_lat, storm_lon, out_path):
    m = folium.Map(location=[storm_lat, storm_lon], zoom_start=5, tiles="cartodbpositron")

    # Build heatmap points -- subsample if the grid is very large, since
    # folium's HeatMap can get slow with hundreds of thousands of points.
    # Ocean cells are skipped entirely (land_mask + NaN check) so the
    # basemap's own water color shows through instead of a heat color.
    heat_data = []
    step = 1  # increase (e.g. 2 or 3) to subsample a very fine grid
    for i in range(0, len(lats), step):
        for j in range(0, len(lons), step):
            if not land_mask[i, j]:
                continue
            val = damage_grid[i, j]
            if np.isnan(val):
                continue
            if val > 0.05:  # skip near-zero cells to keep the file lighter
                heat_data.append([lats[i], lons[j], float(val)])

    HeatMap(heat_data, radius=8, blur=10, max_zoom=6).add_to(m)

    folium.Marker(
        [storm_lat, storm_lon],
        popup="Storm center",
        icon=folium.Icon(color="black", icon="info-sign"),
    ).add_to(m)

    m.save(out_path)
    print(f"Saved interactive map to {out_path}")


# ==========================================================================
# Step 9: Google Maps frontend export (transparent overlay + manifest)
# ==========================================================================


def export_ground_overlay(
    damage_grid: np.ndarray,
    out_png_path: Path,
    min_alpha: int = 40,
    max_alpha: int = 210,
) -> None:
    """Renders damage_grid as a transparent PNG for Google Maps'
    GroundOverlay: same color ramp as plot_static_map, but with an alpha
    channel instead of an opaque background.

    - Ocean / NaN cells -> fully transparent (alpha=0), so the base map's
      own water tiles show through untouched.
    - Land cells fade in proportional to risk (min_alpha at damage=0,
      max_alpha at damage=1), so low-risk land doesn't visually flood
      the map the same as high-risk land.
    """
    from PIL import Image

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "risk", ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e0000"]
    )
    valid = ~np.isnan(damage_grid)
    clipped = np.clip(np.nan_to_num(damage_grid, nan=0.0), 0, 1)

    rgba = (cmap(clipped) * 255).astype(np.uint8)  # (H, W, 4)
    alpha = np.where(
        valid,
        (min_alpha + clipped * (max_alpha - min_alpha)).astype(np.uint8),
        0,
    ).astype(np.uint8)
    rgba[..., 3] = alpha

    # build_grid's row 0 = lat_min (south). Standard raster images are
    # north-up (row 0 = top = north), so flip vertically to match, or the
    # overlay will render upside down on the map.
    rgba = np.flipud(rgba)

    out_png_path = Path(out_png_path)
    out_png_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(out_png_path)
    print(f"Saved ground overlay PNG to {out_png_path}")


def export_web_bundle(
    damage_grid: np.ndarray,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    storm: dict,
    out_dir: str | Path = WEB_OUTPUT_DIR,
) -> None:
    """Writes everything index.html needs to render the overlay:
      - overlay.png  : transparent risk-color raster (see above)
      - overlay.json : the exact lat/lon bounds for that PNG, in the shape
        google.maps.LatLngBounds expects, plus the storm's own parameters
        - so nothing is hardcoded in the JS, it all comes from this run.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    export_ground_overlay(damage_grid, out_dir / "overlay.png")

    def _clean(v):
        if isinstance(v, float) and np.isnan(v):
            return None
        return v

    manifest = {
        "bounds": {
            "south": lat_min,
            "west": lon_min,
            "north": lat_max,
            "east": lon_max,
        },
        "storm": {k: _clean(v) for k, v in storm.items()},
        "max_damage": float(np.nanmax(damage_grid)),
    }
    json_path = out_dir / "overlay.json"
    with open(json_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved overlay manifest to {json_path}")


# ==========================================================================
# Main
# ==========================================================================


def main(LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, STORM, GRID_RESO=0.1):
    print("Building target grid...")
    lats, lons, lat_grid, lon_grid = build_grid(
        LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, GRID_RESO
    )
    print(f"Grid shape: {lat_grid.shape} ({lat_grid.size} cells)")

    print("Computing distance from storm center to every cell...")
    distance_grid = haversine_distance_km(STORM["lat"], STORM["lon"], lat_grid, lon_grid)

    print("Computing wind field (calibrated against rmw/r34avg/r50avg/r64avg, "
          "outer decay shaped by mslp)...")
    B = fit_holland_B(STORM["vmax"], STORM["mslp"])
    wind_grid = build_calibrated_wind_field(distance_grid, STORM, B=B)

    print("Loading and resampling elevation...")
    elevation_grid = load_elevation_on_grid(ELEVATION_NC_PATH, lats, lons)
    land_mask = elevation_grid >= 0  # True = land, False = ocean

    print("Loading and resampling population density...")
    population_grid = load_population_on_grid(POPULATION_TIF_PATH, lats, lons)

    print("Computing composite damage index...")
    damage_grid = compute_damage_index(
        wind_grid, population_grid, elevation_grid, WEIGHTS,
        land_mask=land_mask, distance_km=distance_grid, decay_km=DISTANCE_DECAY_KM,
    )

    print("Exporting Google Maps overlay bundle...")
    export_web_bundle(
        damage_grid, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, STORM,
        out_dir=WEB_OUTPUT_DIR,
    )

    print("Rendering maps...")
    plot_static_map(
        damage_grid, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
        STORM["lat"], STORM["lon"], OUTPUT_PNG,
    )
    plot_interactive_map(damage_grid, lats, lons, land_mask, STORM["lat"], STORM["lon"], OUTPUT_HTML)

    print("\nDone. Highest-risk cell value:", float(np.nanmax(damage_grid)))
    print(f"Google Maps bundle written to ./{WEB_OUTPUT_DIR}/ "
          f"(overlay.png + overlay.json) - copy index.html in there and serve it.")