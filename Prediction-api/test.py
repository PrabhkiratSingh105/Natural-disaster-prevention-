from pathlib import Path

import torch
from predict import run_prediction
import Damage

PREDICTION_DIR = Path(__file__).resolve().parent

code: dict = {'BIPARJOY': 'data/testData/3DIMG_05JUN2023_2200_L1C_ASIA_MER_V01R00.h5',
              'FANI': 'data/testData/3DIMG_28APR2019_2130_L1C_ASIA_MER_V01R00.h5',
              'TAUKTAE': 'data/testData/3DIMG_15MAY2021_2330_L1C_ASIA_MER_V01R00.h5'}

# (lat, lon) as floats, not strings
grd_point: dict = {'FANI': (19.78, 85.81), 'BIPARJOY': (23.21, 68.61),
                    'TAUKTAE': (20.82, 71.03)}

era_data:dict = {'FANI': 'data/testData/FANI_2019116N02090_201904.nc', 'BIPARJOY':
                'data/testData/BIPARJOY_2023156N10067_202306.nc',
                 'TAUKTAE': None}


def create_damageMap(s: str, LAT_MIN, LAT_MAX, LONG_MIN, LONG_MAX, GRID_RESO):
    global results
    results = run_prediction(
        checkpoint=r"runs\india_finetune\best_model_finetuned.pt",
        h5_path=f"{code[s]}",
        storm_name=s,
        era5_file=era_data[s],
        named_csv=r"data\final_ouyputNamed.csv",
        unnamed_csv=r"data\finat_outputUnamed.xlsx",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    STORM = {
        'vmax': results[0]['vmax'],
        'rmw': results[0]['rmw'],          # was 'rmv' - Damage.py reads storm["rmw"]
        'r34avg': results[0]['r34avg'],
        'mslp': results[0]['mslp'],
        'r50avg': results[0]['r50avg'],
        'r64avg': results[0]['r64avg'],
        'lat': grd_point[s][0],            # index 0 = lat
        'lon': grd_point[s][1],            # index 1 = lon
    }

    Damage.main(LAT_MIN=LAT_MIN, LAT_MAX=LAT_MAX, LON_MIN=LONG_MIN, LON_MAX=LONG_MAX,
                GRID_RESO=GRID_RESO, STORM=STORM)


def predict_for_area(extreme_points, center):
    """Run the trained model for a selected area and return frontend values.

    The model predicts storm intensity and radii, not a new storm location.
    Therefore the selected area's center is retained as the heatmap center,
    while the selected extreme points define the prediction grid.
    """
    if not extreme_points:
        extreme_points = {
            "north": {"lat": float(center["lat"]) + 5, "lon": float(center["lon"])},
            "south": {"lat": float(center["lat"]) - 5, "lon": float(center["lon"])},
            "east": {"lat": float(center["lat"]), "lon": float(center["lon"]) + 5},
            "west": {"lat": float(center["lat"]), "lon": float(center["lon"]) - 5},
        }

    north = extreme_points["north"]
    south = extreme_points["south"]
    east = extreme_points["east"]
    west = extreme_points["west"]

    cyclone = center.get("cyclone", "FANI")
    if cyclone not in code:
        raise ValueError(f"Unsupported cyclone: {cyclone}")

    checkpoint = PREDICTION_DIR / "runs/india_finetune/best_model_finetuned.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint}. "
            "Train the model first or place best_model_finetuned.pt in "
            "Prediction-api/runs/india_finetune/."
        )

    results = run_prediction(
        checkpoint=str(checkpoint),
        h5_path=str(PREDICTION_DIR / code[cyclone]),
        storm_name=cyclone,
        era5_file=str(PREDICTION_DIR / era_data[cyclone]) if era_data[cyclone] else None,
        named_csv=str(PREDICTION_DIR / "data/final_ouyputNamed.csv"),
        unnamed_csv=str(PREDICTION_DIR / "data/finat_outputUnamed.xlsx"),
        lat=float(center["lat"]),
        lon=float(center["lon"]),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    prediction = results[0]

    return {
        "cyclone": cyclone,
        "center": {"lat": float(center["lat"]), "lon": float(center["lon"])},
        "radius_meters": float(prediction["r34avg"]) * 1852.0,
        "radius_nautical_miles": float(prediction["r34avg"]),
        "intensity": prediction,
        "extreme_points": extreme_points,
        "grid": {
            "north_lat": float(north["lat"]),
            "south_lat": float(south["lat"]),
            "east_lon": float(east["lon"]),
            "west_lon": float(west["lon"]),
        },
    }


if __name__ == "__main__":
    create_damageMap("FANI", LAT_MIN=8.4, LAT_MAX=34.4, LONG_MIN=68.7, LONG_MAX=97, GRID_RESO=0.1)