"""
train_finetune.py

Fine-tunes the pretrained TCIRModel onto INSAT-3D/3DR India-basin data.

BEFORE FIRST REAL RUN:
    python train_finetune.py --inspect-only --root-dir /path/to/data \
        --named-csv final_ouyputNamed.csv --unnamed-csv finat_outputUnamed.xlsx \
        --era5-dir /path/to/data/era5_downloads

This prints the actual on-disk structure (h5 keys/shapes, csv columns,
nc variables) so you can confirm the ASSUMPTION-tagged constants below
before training starts. Do not run full training until that output looks
right for your data - several of the constants below were inferred from a
single sample file and a compressed environment that did not have h5py /
netCDF4 / torch available, so they were NOT executed end-to-end here.

Known confirmed-by-inspection facts (from the uploaded sample files):
  - INSAT h5 raw channel datasets are named IMG_<CH> for
    CH in {VIS, SWIR, MIR, TIR1, TIR2, WV}, each with companion
    calibration lookup datasets IMG_<CH>_RADIANCE (all channels),
    IMG_<CH>_TEMP (MIR/TIR1/TIR2/WV) or IMG_<CH>_ALBEDO (VIS).
  - The ERA5 .nc file is a CDS/GRIB-derived NetCDF4 with short variable
    names u10, v10, msl, sst and a "valid_time" (not "time") coordinate,
    plus "latitude"/"longitude" and an extra "number"/"realization" dim.
  - The label CSVs (final_ouyputNamed.csv, finat_outputUnamed.xlsx) are
    ONE ROW PER STORM (not per frame) - confirmed with the user, who
    said the same storm-level label applies to every frame of that storm.

ASSUMPTIONS not independently confirmed against the real directory tree
(only seen in a screenshot) - re-check with --inspect-only:
  - Named-storm frames live under <root>/strongStroms4/stromName/<STORM>/*.h5
  - Unnamed-storm frames live under <root>/weakName/<STORM>/*.h5
  - Every frame's timestamp can be parsed from its filename.
  - Whether to use raw counts or calibrated TEMP/RADIANCE/ALBEDO values
    for the model input (this script defaults to calibrated values where
    a LUT dataset exists, raw counts otherwise) - confirm this is the
    intended physical unit before trusting the pretrained stem transfer.
"""

from __future__ import annotations

import argparse
import csv as csv_module
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model_finetune import (
    INSAT_RAW_ORDER,
    MODEL_CHANNEL_ORDER,
    TCIRFineTuneModel,
    build_and_load_finetune_model,
    reorder_insat_channels,
)

HEADS = ["vmax", "mslp", "r34avg", "rmw", "r50avg", "r64avg"]
ALWAYS_PRESENT_HEADS = {"vmax", "mslp", "r34avg"}
ERA5_VARS = ["u10", "v10", "msl", "sst"]  # confirmed short names in the sample .nc


# --------------------------------------------------------------------------
# Diagnostics (spec: "before you write anything" - re-run any time the
# real directory tree differs from what's assumed above)
# --------------------------------------------------------------------------

def inspect_directory(root_dir: Path, max_entries: int = 5) -> None:
    print(f"\n--- Directory tree under {root_dir} (top levels) ---")
    for p in sorted(root_dir.iterdir()):
        print(f"  {p}")
        if p.is_dir():
            children = sorted(p.iterdir())[:max_entries]
            for c in children:
                print(f"    {c}")


def inspect_h5_sample(h5_path: Path) -> None:
    try:
        import h5py
    except ImportError:
        print(f"[inspect] h5py not installed - cannot open {h5_path}. "
              f"pip install h5py and re-run --inspect-only.")
        return
    print(f"\n--- HDF5 structure: {h5_path} ---")
    with h5py.File(h5_path, "r") as f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}: shape={obj.shape} dtype={obj.dtype}")
        f.visititems(visitor)
        print("  Root attrs:", dict(f.attrs))


def inspect_labels_csv(csv_path: Path) -> None:
    print(f"\n--- Labels file: {csv_path} ---")
    df = pd.read_excel(csv_path) if csv_path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(csv_path)
    print("  columns:", list(df.columns))
    print(df.head())


def inspect_era5_nc(nc_path: Path) -> None:
    try:
        import xarray as xr
    except ImportError:
        print(f"[inspect] xarray not installed - cannot open {nc_path}. "
              f"pip install xarray netCDF4 and re-run --inspect-only.")
        return
    print(f"\n--- NetCDF structure: {nc_path} ---")
    ds = xr.open_dataset(nc_path)
    print(ds)


# --------------------------------------------------------------------------
# Label loading (storm-level rows, broadcast to every frame of that storm)
# --------------------------------------------------------------------------

_COLUMN_ALIASES = {
    "rmw": "rmw", "rmv": "rmw",
    "r34avg": "r34avg",
    "r50avg": "r50avg",
    "r64avg": "r64avg",
    "vmax": "vmax",
    "mslp": "mslp",
    "name": "name",
    "longitude": "longitude", "long": "longitude",
    "latitude": "latitude",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for c in df.columns:
        key = c.strip().lower()
        if key in _COLUMN_ALIASES:
            rename[c] = _COLUMN_ALIASES[key]
    return df.rename(columns=rename)


def _normalize_storm_key(name: str) -> str:
    """Strip punctuation/whitespace/case so folder names and CSV NAME
    values match even when one has separators the other doesn't
    (e.g. 'GULAB:SHAHEEN-GU' vs a folder called 'GULABSHAHEEN-GU')."""
    return re.sub(r"[^A-Za-z0-9]", "", str(name)).upper()


_LATLON_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*([NSEW])?\s*$")


def _parse_latlon_token(raw) -> Tuple[Optional[str], float]:
    """Parses a single geo value like '22.9N', '68.8E', '-13.2', or a bare
    float, and returns (axis, signed_value) where axis is 'lat' if the
    suffix was N/S, 'lon' if it was E/W, or None if there was no
    compass-direction suffix to tell us which axis it belongs to.

    This is needed because the label CSVs are NOT consistent about which
    column ('longitude' vs 'latitude') actually holds the longitude vs
    latitude value - e.g. in final_ouyputNamed.csv the first row has
    longitude='68E', latitude='22.9N' (correct order) but the next row
    has longitude='10N', latitude='63.8E' (swapped). We can't trust the
    column header, only the compass letter on the value itself."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None, float("nan")
    s = str(raw).strip().upper()
    m = _LATLON_RE.match(s)
    if not m:
        return None, float("nan")
    val = float(m.group(1))
    suffix = m.group(2)
    if suffix in ("S", "W"):
        val = -val
    if suffix in ("N", "S"):
        return "lat", val
    if suffix in ("E", "W"):
        return "lon", val
    return None, val  # no compass letter - can't tell which axis this is


def load_storm_labels(named_csv: Path, unnamed_csv: Path) -> Dict[str, dict]:
    """Returns {normalized_storm_key: {head: value, ...}} for every storm
    in both label files. Values are float, NaN where missing."""
    named = _normalize_columns(pd.read_csv(named_csv))
    unnamed = _normalize_columns(
        pd.read_excel(unnamed_csv) if unnamed_csv.suffix.lower() in (".xlsx", ".xls")
        else pd.read_csv(unnamed_csv)
    )

    labels: Dict[str, dict] = {}
    unresolved_latlon: List[str] = []
    for df in (named, unnamed):
        if "name" not in df.columns:
            raise ValueError(f"Expected a NAME column, got: {list(df.columns)}")
        for _, row in df.iterrows():
            if pd.isna(row.get("name")):
                continue
            key = _normalize_storm_key(row["name"])
            labels[key] = {h: (float(row[h]) if h in df.columns and pd.notna(row.get(h)) else float("nan"))
                            for h in HEADS}

            # lat/lon: don't trust which column is which (see _parse_latlon_token
            # docstring) - parse both candidate columns by compass letter and
            # route each value to whichever axis it actually is.
            lat_val, lon_val = float("nan"), float("nan")
            for col in ("longitude", "latitude"):
                if col not in df.columns:
                    continue
                axis, val = _parse_latlon_token(row.get(col))
                if axis == "lat":
                    lat_val = val
                elif axis == "lon":
                    lon_val = val
            labels[key]["lat"] = lat_val
            labels[key]["lon"] = lon_val
            if np.isnan(lat_val) or np.isnan(lon_val):
                unresolved_latlon.append(row["name"])

    if unresolved_latlon:
        print(f"WARNING: could not resolve lat/lon (no N/S/E/W suffix found, or "
              f"missing column) for {len(unresolved_latlon)} storm(s): "
              f"{unresolved_latlon}. ERA5 metadata will be zero for every frame "
              f"of these storms.")
    return labels


# --------------------------------------------------------------------------
# INSAT frame reading
# --------------------------------------------------------------------------

def read_insat_frame(h5_path: Path) -> np.ndarray:
    """Reads one INSAT .h5 frame and returns a (6, H, W) float32 array in
    MODEL_CHANNEL_ORDER, using calibrated physical values where a lookup
    table exists (TEMP for thermal channels, ALBEDO for VIS, RADIANCE as
    fallback), raw counts otherwise.

    ASSUMPTION (unverified - re-check with inspect_h5_sample): raw counts
    are stored as an (1, H, W) or (H, W) integer dataset "IMG_<CH>", and
    IMG_<CH>_TEMP / IMG_<CH>_ALBEDO / IMG_<CH>_RADIANCE are 1-D lookup
    tables indexed by the raw count value.
    """
    import h5py

    channels = {}
    with h5py.File(h5_path, "r") as f:
        for ch in INSAT_RAW_ORDER:
            raw_key = f"IMG_{ch}"
            if raw_key not in f:
                raise KeyError(f"{h5_path}: expected dataset '{raw_key}' not found. "
                                f"Available: {list(f.keys())}")
            raw = np.array(f[raw_key])
            raw = raw.squeeze()  # drop leading singleton dim if present

            lut = None
            for suffix in ("_TEMP", "_ALBEDO", "_RADIANCE"):
                lut_key = raw_key + suffix
                if lut_key in f:
                    lut = np.array(f[lut_key]).squeeze()
                    break

            if lut is not None:
                idx = np.clip(raw.astype(np.int64), 0, len(lut) - 1)
                channels[ch] = lut[idx].astype(np.float32)
            else:
                channels[ch] = raw.astype(np.float32)

    stack = np.stack([channels[ch] for ch in INSAT_RAW_ORDER], axis=0)  # (6, H, W) in INSAT_RAW_ORDER
    stack_t = torch.from_numpy(stack)
    stack_t = reorder_insat_channels(stack_t)  # -> MODEL_CHANNEL_ORDER
    return stack_t.numpy()


def parse_timestamp_from_filename(path: Path) -> Optional[pd.Timestamp]:
    """ASSUMPTION: filenames look like 3DIMG_26APR2019_0000_L1C_....h5
    (ddMONyyyy_HHMM). Adjust this regex if the real frame filenames differ."""
    m = re.search(r"(\d{2}[A-Z]{3}\d{4})_(\d{4})", path.stem.upper())
    if not m:
        return None
    date_str, time_str = m.groups()
    try:
        return pd.to_datetime(f"{date_str} {time_str}", format="%d%b%Y %H%M")
    except ValueError:
        return None


# --------------------------------------------------------------------------
# ERA5 point extraction
# --------------------------------------------------------------------------

class ERA5Lookup:
    """Lazily opens per-storm ERA5 .nc files and point-extracts the 4
    scalar features nearest to a given (lat, lon, time)."""

    def __init__(self, era5_dir: Path, tolerance_minutes: int = 45):
        self.era5_dir = era5_dir
        self.tolerance = pd.Timedelta(minutes=tolerance_minutes)
        self._cache: Dict[str, "xr.Dataset"] = {}  # noqa: F821

    def _dataset_for_storm(self, storm_key: str):
        import xarray as xr

        if storm_key in self._cache:
            return self._cache[storm_key]
        # ASSUMPTION: filenames are <STORMNAME>_<SID>_<YYYYMM>.nc - one or
        # more monthly files per storm. Glob and concat along time.
        matches = [p for p in self.era5_dir.glob("*.nc")
                   if _normalize_storm_key(p.stem.split("_")[0]) == storm_key]
        if not matches:
            self._cache[storm_key] = None
            return None
        datasets = [xr.open_dataset(p) for p in matches]
        ds = xr.concat(datasets, dim="valid_time") if len(datasets) > 1 else datasets[0]
        self._cache[storm_key] = ds
        return ds

    def extract(self, storm_key: str, lat: float, lon: float, ts: pd.Timestamp) -> np.ndarray:
        ds = self._dataset_for_storm(storm_key)
        if ds is None:
            return np.full(len(ERA5_VARS), np.nan, dtype=np.float32)
        try:
            point = ds.sel(latitude=lat, longitude=lon, method="nearest")
            point = point.sel(valid_time=ts, method="nearest", tolerance=self.tolerance)
            values = [float(point[v].values) for v in ERA5_VARS]
        except (KeyError, ValueError):
            values = [float("nan")] * len(ERA5_VARS)
        return np.array(values, dtype=np.float32)


# --------------------------------------------------------------------------
# Frame index
# --------------------------------------------------------------------------

@dataclass
class FrameEntry:
    h5_path: Path
    storm_key: str
    is_named: bool
    timestamp: Optional[pd.Timestamp]
    lat: float
    lon: float


def _storm_latlon(key: str, storm_labels: Dict[str, dict]) -> Tuple[float, float]:
    info = storm_labels.get(key)
    if info is None:
        return float("nan"), float("nan")
    return info.get("lat", float("nan")), info.get("lon", float("nan"))


def _h5_is_readable(h5_path: Path) -> bool:
    """Cheap corruption check - opens the file and walks its structure
    without loading full arrays. Catches truncated/incomplete downloads
    (see: OSError 'truncated file: eof=... stored_eof=...') before they
    can crash a DataLoader worker mid-epoch."""
    import h5py
    try:
        with h5py.File(h5_path, "r") as f:
            f.visititems(lambda name, obj: None)
        return True
    except OSError:
        return False


def build_frame_index(root_dir: Path, storm_labels: Dict[str, dict]) -> List[FrameEntry]:
    """ASSUMPTION: named storms under <root>/strongStroms4/stromName/<STORM>/*.h5,
    unnamed under <root>/weakName/<STORM>/*.h5. Re-check with --inspect-only
    and adjust these two globs if the real layout differs.

    lat/lon are looked up per storm from storm_labels (parsed in
    load_storm_labels) rather than hardcoded to NaN, so ERA5Lookup can
    actually find the nearest grid point instead of always falling back
    to an all-zero metadata vector.

    Each .h5 is opened once here to check it isn't truncated/corrupt;
    unreadable files are excluded (with a summary warning) instead of
    crashing training later, since a single bad download shouldn't take
    down an entire run hours in."""
    entries: List[FrameEntry] = []
    missing_latlon_keys: set = set()
    corrupt_files: List[Path] = []

    named_root = root_dir / "strongStroms4" / "stromName"
    if named_root.exists():
        for storm_dir in sorted(named_root.iterdir()):
            if not storm_dir.is_dir():
                continue
            key = _normalize_storm_key(storm_dir.name)
            lat, lon = _storm_latlon(key, storm_labels)
            if np.isnan(lat) or np.isnan(lon):
                missing_latlon_keys.add(storm_dir.name)
            for h5_path in sorted(storm_dir.glob("*.h5")):
                if not _h5_is_readable(h5_path):
                    corrupt_files.append(h5_path)
                    continue
                entries.append(FrameEntry(h5_path, key, True, parse_timestamp_from_filename(h5_path), lat, lon))

    unnamed_root = root_dir / "weakName"
    if unnamed_root.exists():
        for storm_dir in sorted(unnamed_root.iterdir()):
            if not storm_dir.is_dir():
                continue
            key = _normalize_storm_key(storm_dir.name)
            lat, lon = _storm_latlon(key, storm_labels)
            if np.isnan(lat) or np.isnan(lon):
                missing_latlon_keys.add(storm_dir.name)
            for h5_path in sorted(storm_dir.glob("*.h5")):
                if not _h5_is_readable(h5_path):
                    corrupt_files.append(h5_path)
                    continue
                entries.append(FrameEntry(h5_path, key, False, parse_timestamp_from_filename(h5_path), lat, lon))

    if not entries:
        raise RuntimeError(
            f"No .h5 frames found under {named_root} or {unnamed_root}. "
            f"Run with --inspect-only and fix build_frame_index()'s glob patterns."
        )
    if missing_latlon_keys:
        print(f"WARNING: {len(missing_latlon_keys)} storm folder(s) have no resolvable "
              f"lat/lon (folder name didn't match any label-CSV storm, or that storm's "
              f"lat/lon couldn't be parsed): {sorted(missing_latlon_keys)}. Every frame "
              f"for these storms will get a zero ERA5 metadata vector.")
    if corrupt_files:
        print(f"WARNING: skipped {len(corrupt_files)} unreadable/truncated .h5 file(s) "
              f"(re-download these when convenient):")
        for p in corrupt_files:
            print(f"  {p}")
    return entries


def storm_level_split(entries: List[FrameEntry], val_fraction: float = 0.2, seed: int = 42
                       ) -> Tuple[List[FrameEntry], List[FrameEntry]]:
    """Splits at the storm level, keeping named/unnamed mixed on both sides."""
    rng = random.Random(seed)

    named_storms = sorted({e.storm_key for e in entries if e.is_named})
    unnamed_storms = sorted({e.storm_key for e in entries if not e.is_named})
    rng.shuffle(named_storms)
    rng.shuffle(unnamed_storms)

    def split(storms: List[str]) -> Tuple[set, set]:
        n_val = max(1, round(len(storms) * val_fraction)) if storms else 0
        return set(storms[n_val:]), set(storms[:n_val])

    named_train, named_val = split(named_storms)
    unnamed_train, unnamed_val = split(unnamed_storms)
    train_storms = named_train | unnamed_train
    val_storms = named_val | unnamed_val

    train = [e for e in entries if e.storm_key in train_storms]
    val = [e for e in entries if e.storm_key in val_storms]
    return train, val


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

class TCIRIndiaDataset(Dataset):
    def __init__(
        self,
        entries: List[FrameEntry],
        storm_labels: Dict[str, dict],
        era5_lookup: ERA5Lookup,
        label_stats: Dict[str, Tuple[float, float]],
        era5_stats: Tuple[np.ndarray, np.ndarray],
    ):
        self.entries = entries
        self.storm_labels = storm_labels
        self.era5_lookup = era5_lookup
        self.label_mean = {h: label_stats[h][0] for h in HEADS}
        self.label_std = {h: label_stats[h][1] for h in HEADS}
        self.era5_mean, self.era5_std = era5_stats

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        image = read_insat_frame(entry.h5_path)  # (6, H, W), MODEL_CHANNEL_ORDER

        raw_meta = self.era5_lookup.extract(entry.storm_key, entry.lat, entry.lon, entry.timestamp)
        meta = (raw_meta - self.era5_mean) / np.where(self.era5_std == 0, 1.0, self.era5_std)
        meta = np.nan_to_num(meta, nan=0.0)  # missing ERA5 point -> zero after standardization

        raw_labels = self.storm_labels.get(entry.storm_key, {h: float("nan") for h in HEADS})
        targets, masks = {}, {}
        for h in HEADS:
            v = raw_labels[h]
            if np.isnan(v):
                targets[h] = 0.0
                masks[h] = 0.0
            else:
                targets[h] = (v - self.label_mean[h]) / (self.label_std[h] if self.label_std[h] != 0 else 1.0)
                masks[h] = 1.0

        return {
            "image": torch.from_numpy(image).float(),
            "meta": torch.from_numpy(meta).float(),
            "targets": {h: torch.tensor(targets[h], dtype=torch.float32) for h in HEADS},
            "masks": {h: torch.tensor(masks[h], dtype=torch.float32) for h in HEADS},
        }


def compute_label_stats(entries: List[FrameEntry], storm_labels: Dict[str, dict]) -> Dict[str, Tuple[float, float]]:
    """Mean/std per head over TRAIN-split values only, computed from the
    valid (non-NaN) values only for the sparse heads."""
    stats = {}
    storm_keys_in_split = {e.storm_key for e in entries}
    for h in HEADS:
        vals = [storm_labels[k][h] for k in storm_keys_in_split
                if k in storm_labels and not np.isnan(storm_labels[k][h])]
        if not vals:
            stats[h] = (0.0, 1.0)
            continue
        arr = np.array(vals, dtype=np.float64)
        std = arr.std()
        stats[h] = (float(arr.mean()), float(std if std > 1e-6 else 1.0))
    return stats


def compute_era5_stats(entries: List[FrameEntry], era5_lookup: ERA5Lookup) -> Tuple[np.ndarray, np.ndarray]:
    samples = []
    for e in entries:
        v = era5_lookup.extract(e.storm_key, e.lat, e.lon, e.timestamp)
        if not np.any(np.isnan(v)):
            samples.append(v)
    if not samples:
        return np.zeros(len(ERA5_VARS), dtype=np.float32), np.ones(len(ERA5_VARS), dtype=np.float32)
    arr = np.stack(samples, axis=0)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


# --------------------------------------------------------------------------
# Masked multi-task loss
# --------------------------------------------------------------------------

class MaskedMultiTaskHuberLoss(nn.Module):
    def __init__(self, weights: Optional[Dict[str, float]] = None, beta: float = 1.0):
        super().__init__()
        # Default: all heads weighted equally. If you want to down-weight
        # the sparse heads (rmw/r50avg/r64avg), pass explicit weights -
        # the spec says to ask rather than guess a scheme, so this
        # defaults to 1.0 for all six.
        self.weights = weights or {h: 1.0 for h in HEADS}
        self.beta = beta

    def forward(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor],
                masks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        total = torch.zeros((), device=next(iter(preds.values())).device)
        for h in HEADS:
            pred = preds[h].squeeze(-1)
            target = targets[h]
            mask = masks[h]
            n_valid = mask.sum()
            if n_valid.item() == 0:
                losses[h] = torch.zeros((), device=pred.device)
                continue
            per_sample = nn.functional.smooth_l1_loss(pred, target, beta=self.beta, reduction="none")
            losses[h] = (per_sample * mask).sum() / n_valid
            total = total + self.weights[h] * losses[h]
        losses["total"] = total
        return losses


# --------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, scaler, device, train: bool,
              log_every: int = 20, tag: str = "", grad_accum_steps: int = 1) -> Dict[str, float]:
    model.train(mode=train)
    totals = {h: 0.0 for h in HEADS + ["total"]}
    n_batches = 0
    n_total_batches = len(loader)
    grad_accum_steps = max(1, grad_accum_steps)

    if train:
        optimizer.zero_grad(set_to_none=True)

    grad_ctx = torch.enable_grad() if train else torch.no_grad()
    with grad_ctx:
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            meta = batch["meta"].to(device, non_blocking=True)
            targets = {h: batch["targets"][h].to(device) for h in HEADS}
            masks = {h: batch["masks"][h].to(device) for h in HEADS}

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                preds = model(image, meta)
                losses = criterion(preds, targets, masks)

            if train:
                loss_to_backward = losses["total"] / grad_accum_steps
                if scaler is not None:
                    scaler.scale(loss_to_backward).backward()
                else:
                    loss_to_backward.backward()

                is_accum_boundary = (n_batches + 1) % grad_accum_steps == 0 or (n_batches + 1) == n_total_batches
                if is_accum_boundary:
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            for k in totals:
                totals[k] += float(losses[k].detach().cpu())
            n_batches += 1

            if log_every and (n_batches % log_every == 0 or n_batches == n_total_batches):
                running_total = totals["total"] / n_batches
                print(f"    [{tag}] batch {n_batches}/{n_total_batches}  "
                      f"running_total_loss={running_total:.4f}", flush=True)

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=Path, required=True, help="Root data folder (contains strongStroms4/, weakName/, era5_downloads/)")
    parser.add_argument("--named-csv", type=Path, required=True)
    parser.add_argument("--unnamed-csv", type=Path, required=True)
    parser.add_argument("--era5-dir", type=Path, required=True)
    parser.add_argument("--pretrained-ckpt", type=Path, required=True, help="Path to best_model.pt")
    parser.add_argument("--output-dir", type=Path, default=Path("./finetune_output"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr-low", type=float, default=1e-4)
    parser.add_argument("--lr-high", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--inspect-only", action="store_true",
                         help="Print directory/h5/csv/nc structure and exit without training.")
    args = parser.parse_args()

    if args.inspect_only:
        inspect_directory(args.root_dir)
        inspect_labels_csv(args.named_csv)
        inspect_labels_csv(args.unnamed_csv)
        sample_h5 = next(args.root_dir.rglob("*.h5"), None)
        if sample_h5:
            inspect_h5_sample(sample_h5)
        sample_nc = next(args.era5_dir.glob("*.nc"), None)
        if sample_nc:
            inspect_era5_nc(sample_nc)
        return

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Labels, frames, split ---
    storm_labels = load_storm_labels(args.named_csv, args.unnamed_csv)
    entries = build_frame_index(args.root_dir, storm_labels)
    train_entries, val_entries = storm_level_split(entries, val_fraction=0.2, seed=args.seed)
    print(f"Frames: {len(entries)} total | {len(train_entries)} train | {len(val_entries)} val", flush=True)

    # --- Stats (train-split only, valid values only for sparse heads) ---
    label_stats = compute_label_stats(train_entries, storm_labels)
    era5_lookup = ERA5Lookup(args.era5_dir)
    era5_stats = compute_era5_stats(train_entries, era5_lookup)
    print("Label stats (mean, std):", label_stats, flush=True)

    train_ds = TCIRIndiaDataset(train_entries, storm_labels, era5_lookup, label_stats, era5_stats)
    val_ds = TCIRIndiaDataset(val_entries, storm_labels, era5_lookup, label_stats, era5_stats)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device == "cuda"))

    # --- Model + weight transfer ---
    model, ckpt = build_and_load_finetune_model(str(args.pretrained_ckpt), meta_dim=len(ERA5_VARS), device=device)
    model.to(device)

    param_groups = model.get_param_groups(lr_low=args.lr_low, lr_high=args.lr_high)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    criterion = MaskedMultiTaskHuberLoss()

    history_path = args.output_dir / "history.csv"
    with open(history_path, "w", newline="") as f:
        writer = csv_module.writer(f)
        writer.writerow(["epoch"] + [f"train_{h}" for h in HEADS + ["total"]] + [f"val_{h}" for h in HEADS + ["total"]])

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        print(f"\n=== epoch {epoch}/{args.epochs} ===", flush=True)

        train_metrics = run_epoch(model, train_loader, criterion, optimizer, scaler, device,
                                   train=True, tag=f"epoch {epoch} train", grad_accum_steps=args.grad_accum_steps)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer=None, scaler=None, device=device,
                                 train=False, tag=f"epoch {epoch} val")
        scheduler.step(val_metrics["total"])
        epoch_secs = time.time() - epoch_start

        per_head = "  ".join(f"{h}(tr={train_metrics[h]:.3f},val={val_metrics[h]:.3f})" for h in HEADS)
        print(f"[epoch {epoch}] train_total={train_metrics['total']:.4f} "
              f"val_total={val_metrics['total']:.4f}  ({epoch_secs:.1f}s)", flush=True)
        print(f"  per-head: {per_head}", flush=True)
        with open(history_path, "a", newline="") as f:
            writer = csv_module.writer(f)
            writer.writerow([epoch] + [train_metrics[h] for h in HEADS + ["total"]]
                             + [val_metrics[h] for h in HEADS + ["total"]])

        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": best_val_loss,
                "label_mean": {h: train_ds.label_mean[h] for h in HEADS},
                "label_std": {h: train_ds.label_std[h] for h in HEADS},
                "era5_mean": era5_stats[0],
                "era5_std": era5_stats[1],
            }, args.output_dir / "best_model_finetuned.pt")
            print(f"  -> saved new best checkpoint (val_total={best_val_loss:.4f})", flush=True)


if __name__ == "__main__":
    main()