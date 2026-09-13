r"""
predict.py

Run the fine-tuned TCIRFineTuneModel on one or more INSAT-3D/3DR .h5
frames and print all 6 predicted values (vmax, mslp, r34avg, rmw,
r50avg, r64avg) in their original physical units.

Reuses read_insat_frame / ERA5Lookup / parse_timestamp_from_filename /
load_storm_labels / _normalize_storm_key from train_finetune.py, so this
script must live in the same folder as train_finetune.py and
model_finetune.py.

No terminal arguments - everything is set in the CONFIG dict below.
Edit the values there and just run:

    python predict.py

Or import and call run_prediction(...) directly from another script /
notebook with your own keyword arguments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

# --- reuse the exact preprocessing the model was trained with ---
from train_finetune import (
    HEADS,
    ERA5_VARS,
    ERA5Lookup,
    read_insat_frame,
    parse_timestamp_from_filename,
    load_storm_labels,
    _normalize_storm_key,
)
from model_finetune import TCIRFineTuneModel


class SingleFileERA5Lookup:
    """Like ERA5Lookup, but points at ONE specific .nc file directly
    instead of globbing a directory for <STORM>_<SID>_<YYYYMM>.nc. Use
    this when you downloaded a plain whole-day file (e.g. via
    download_era5.py without --storm-name/--sid) rather than a
    per-storm-named one."""

    def __init__(self, nc_path: Path, tolerance_minutes: int = 45):
        nc_path = Path(nc_path)
        if not str(nc_path) or not nc_path.exists():
            raise FileNotFoundError(
                f"era5_file '{nc_path}' does not exist (or is an empty string). "
                f"Check the path you passed in - this is a hard error, not a silent skip, "
                f"because a typo'd path used to silently produce zero-vector metadata."
            )
        import xarray as xr
        self.ds = xr.open_dataset(nc_path)
        self.tolerance = pd.Timedelta(minutes=tolerance_minutes)

        lat_vals = self.ds["latitude"].values
        lon_vals = self.ds["longitude"].values
        time_vals = pd.to_datetime(self.ds["valid_time"].values)
        print(f"[SingleFileERA5Lookup] opened {nc_path.name}: "
              f"lat [{lat_vals.min():.2f}, {lat_vals.max():.2f}], "
              f"lon [{lon_vals.min():.2f}, {lon_vals.max():.2f}], "
              f"time [{time_vals.min()} .. {time_vals.max()}] "
              f"({len(time_vals)} steps)")
        self._lat_range = (float(lat_vals.min()), float(lat_vals.max()))
        self._lon_range = (float(lon_vals.min()), float(lon_vals.max()))
        self._time_range = (time_vals.min(), time_vals.max())

    def extract(self, storm_key: Optional[str], lat: float, lon: float, ts: pd.Timestamp) -> np.ndarray:
        # storm_key is accepted (unused) only so this drop-in-replaces
        # ERA5Lookup.extract's call signature.
        problems = []
        if not (self._lat_range[0] <= lat <= self._lat_range[1]):
            problems.append(f"lat={lat} is outside the file's lat range {self._lat_range}")
        if not (self._lon_range[0] <= lon <= self._lon_range[1]):
            problems.append(f"lon={lon} is outside the file's lon range {self._lon_range}")
        if not (self._time_range[0] - self.tolerance <= ts <= self._time_range[1] + self.tolerance):
            problems.append(f"timestamp={ts} is outside the file's time range "
                             f"{self._time_range} (+/- {self.tolerance} tolerance)")
        if problems:
            print(f"[SingleFileERA5Lookup] extraction will fail for this frame:")
            for p in problems:
                print(f"    - {p}")

        try:
            point = self.ds.sel(latitude=lat, longitude=lon, method="nearest")
            point = point.sel(valid_time=ts, method="nearest", tolerance=self.tolerance)
            values = [float(point[v].values) for v in ERA5_VARS]
        except (KeyError, ValueError) as e:
            print(f"[SingleFileERA5Lookup] extract() raised {type(e).__name__}: {e}")
            values = [float("nan")] * len(ERA5_VARS)
        return np.array(values, dtype=np.float32)


# ============================================================================
# EDIT THESE - this is the only section you should need to touch
# ============================================================================
CONFIG = {
    "checkpoint": r"runs\india_finetune\best_model_finetuned.pt",

    # Exactly ONE of these two - set the other to None.
    "h5_path": r"data\insantData\3DIMG_05JUN2023_0100_L1C_ASIA_MER_V01R00.h5",  # single frame
    "h5_dir": None,   # or e.g. r"data\strongStroms4\stromName\BIPARJOY" for a whole storm

    "storm_name": "BIPARJOY",        # used for ERA5 lookup + optional lat/lon lookup; None to skip
    "lat": None,                     # manual override; None = look up from the CSVs below
    "lon": None,                     # manual override; None = look up from the CSVs below

    # ERA5 metadata source - set AT MOST ONE of these two, the other None:
    "era5_file": None,               # a single .nc file (e.g. from download_era5.py without --storm-name/--sid)
    "era5_dir": r"era5_downloads",   # a folder of <STORM>_<SID>_<YYYYMM>.nc files (per-storm convention)
    # Leave both None to skip ERA5 entirely (meta defaults to zero).

    "named_csv": r"data\final_ouyputNamed.csv",     # only used to resolve lat/lon by storm_name
    "unnamed_csv": r"data\finat_outputUnamed.xlsx",  # only used to resolve lat/lon by storm_name

    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
# ============================================================================


def load_model(checkpoint_path: Path, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = TCIRFineTuneModel(in_channels=6, meta_dim=len(ERA5_VARS))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    label_mean = ckpt["label_mean"]
    label_std = ckpt["label_std"]
    era5_mean = np.asarray(ckpt["era5_mean"], dtype=np.float32)
    era5_std = np.asarray(ckpt["era5_std"], dtype=np.float32)
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
          f"(val_loss={ckpt.get('val_loss', float('nan')):.4f})")
    return model, label_mean, label_std, era5_mean, era5_std


def build_meta_vector(
    h5_path: Path,
    storm_key: Optional[str],
    lat: Optional[float],
    lon: Optional[float],
    era5_lookup,  # ERA5Lookup | SingleFileERA5Lookup | None - both share the same .extract() signature
    era5_mean: np.ndarray,
    era5_std: np.ndarray,
) -> np.ndarray:
    """Mirrors TCIRIndiaDataset.__getitem__'s meta handling: raw ERA5
    extraction, standardize with the TRAIN-set stats from the
    checkpoint, then zero out anything that couldn't be resolved."""
    if era5_lookup is None or storm_key is None or lat is None or lon is None:
        raw_meta = np.full(len(ERA5_VARS), np.nan, dtype=np.float32)
    else:
        ts = parse_timestamp_from_filename(h5_path)
        if ts is None:
            print(f"WARNING: couldn't parse a timestamp out of {h5_path.name}; "
                  f"ERA5 lookup will fail and meta will be zero.")
            raw_meta = np.full(len(ERA5_VARS), np.nan, dtype=np.float32)
        else:
            raw_meta = era5_lookup.extract(storm_key, lat, lon, ts)

    meta = (raw_meta - era5_mean) / np.where(era5_std == 0, 1.0, era5_std)
    meta = np.nan_to_num(meta, nan=0.0)
    if np.isnan(raw_meta).any():
        per_var = ", ".join(
            f"{name}={'NaN->0' if np.isnan(v) else f'{v:.3f}'}"
            for name, v in zip(ERA5_VARS, raw_meta)
        )
        print(f"NOTE: {h5_path.name} - some ERA5 variables unresolved: {per_var}")
    return meta.astype(np.float32)


def predict_one(
    model,
    h5_path: Path,
    meta: np.ndarray,
    label_mean: Dict[str, float],
    label_std: Dict[str, float],
    device: str,
) -> Dict[str, float]:
    image = read_insat_frame(h5_path)  # (6, H, W)
    image_t = torch.from_numpy(image).float().unsqueeze(0).to(device)
    meta_t = torch.from_numpy(meta).float().unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(image_t, meta_t)

    result = {}
    for h in HEADS:
        norm_val = float(preds[h].squeeze().cpu())
        result[h] = norm_val * label_std[h] + label_mean[h]
    return result


def print_prediction(name: str, values: Dict[str, float]):
    print(f"\n{name}")
    print(f"  vmax    (max wind, kt) : {values['vmax']:.2f}")
    print(f"  mslp    (min pressure) : {values['mslp']:.2f}")
    print(f"  r34avg  (34kt radius)  : {values['r34avg']:.2f}")
    print(f"  rmw     (radius max wind): {values['rmw']:.2f}")
    print(f"  r50avg  (50kt radius)  : {values['r50avg']:.2f}")
    print(f"  r64avg  (64kt radius)  : {values['r64avg']:.2f}")


def run_prediction(
    checkpoint: str,
    h5_path: Optional[str] = None,
    h5_dir: Optional[str] = None,
    storm_name: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    era5_file: Optional[str] = None,
    era5_dir: Optional[str] = None,
    named_csv: Optional[str] = None,
    unnamed_csv: Optional[str] = None,
    device: str = "cpu",
) -> List[Dict[str, float]]:
    """Runs the model on one frame (h5_path) or every frame in a folder
    (h5_dir), prints all 6 predictions per frame (+ an average if more
    than one frame), and returns the list of per-frame result dicts.

    ERA5 metadata source: pass era5_file for a single .nc file (e.g. a
    plain whole-day download), or era5_dir for a folder of
    <STORM>_<SID>_<YYYYMM>.nc files (era5_file takes priority if both
    are given). Leave both None to skip ERA5 (meta defaults to zero)."""
    if not h5_path and not h5_dir:
        raise ValueError("Pass either h5_path (single frame) or h5_dir (folder of frames)")
    if era5_file and era5_dir:
        print("NOTE: both era5_file and era5_dir were given - using era5_file, ignoring era5_dir.")

    checkpoint = Path(checkpoint)
    model, label_mean, label_std, era5_mean, era5_std = load_model(checkpoint, device)

    # Resolve lat/lon: manual override wins, otherwise look up from CSVs by storm name.
    storm_key = _normalize_storm_key(storm_name) if storm_name else None
    if (lat is None or lon is None) and storm_key and named_csv and unnamed_csv:
        storm_labels = load_storm_labels(Path(named_csv), Path(unnamed_csv))
        info = storm_labels.get(storm_key)
        if info:
            lat = info.get("lat") if lat is None else lat
            lon = info.get("lon") if lon is None else lon
    if storm_key and (lat is None or lon is None):
        print(f"WARNING: no lat/lon resolved for storm '{storm_name}' - "
              f"set lat/lon manually in CONFIG, or named_csv/unnamed_csv to look it up. "
              f"Proceeding with zero-vector ERA5 metadata.")

    if era5_file:
        era5_lookup = SingleFileERA5Lookup(Path(era5_file))
    elif era5_dir:
        era5_lookup = ERA5Lookup(Path(era5_dir))
    else:
        era5_lookup = None

    h5_files: List[Path] = [Path(h5_path)] if h5_path else sorted(Path(h5_dir).glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found at {h5_dir}")

    all_results: List[Dict[str, float]] = []
    for p in h5_files:
        meta = build_meta_vector(p, storm_key, lat, lon, era5_lookup, era5_mean, era5_std)
        result = predict_one(model, p, meta, label_mean, label_std, device)
        all_results.append(result)
        print_prediction(p.name, result)

    if len(all_results) > 1:
        avg = {h: float(np.mean([r[h] for r in all_results])) for h in HEADS}
        print_prediction(f"AVERAGE over {len(all_results)} frames", avg)

    return all_results


if __name__ == "__main__":
    run_prediction(**CONFIG)