"""
train_tcir.py

Training script for the TCIR-style tropical cyclone intensity regression
model defined in `tcir_model.py`.

Expects:
  - An HDF5 file containing an image dataset (default key "matrix") of
    shape (N, 201, 201, 4)  -- one 4-channel satellite patch per storm
    snapshot, in the SAME row order as the CSV below. This matches the
    standard TCIR benchmark layout (IR / WV / VIS / PMW channels).
  - A CSV file with N rows (same order as the HDF5 file) containing at
    least the three target columns (VMAX, MSLP, R34avg-style radius) and,
    optionally, extra scalar metadata columns (basin one-hot, sin/cos of
    day-of-year, etc.) to feed the model's metadata branch.

If your column names differ from the defaults below, just override them
on the command line -- see `--vmax-col`, `--mslp-col`, `--r34-col`,
`--meta-cols`.

Usage example:
    python train_tcir.py \
        --h5 /data/TCIR-ALL_2017.h5 \
        --csv /data/TCIR-ALL_2017_labels.csv \
        --output-dir ./runs/exp1 \
        --vmax-col Vmax --mslp-col MSLP --r34-col R35_4qAVG \
        --meta-cols basin_sin,basin_cos,doy_sin,doy_cos \
        --epochs 60 --batch-size 32 --lr 3e-4

Place this file in the same directory as `tcir_model.py` (or make sure
tcir_model.py is importable from your PYTHONPATH).

NOTE ON NaN HANDLING (read this if you hit NaN losses):
TCIR satellite patches routinely contain NaN and/or +-inf pixels (most
commonly in the passive-microwave channel, which doesn't cover every
snapshot). This script treats BOTH NaN and inf as "missing" when it
estimates per-channel normalization statistics, falls back to a safe
mean=0/std=1 for any channel that turns out to be entirely missing in
the sampled data, and zero-fills missing pixels *after* normalization
(not before), so a bad/missing pixel can never propagate a NaN or a
huge finite value into the normalized tensor. A sanity check right
after stats are computed will raise immediately, with a clear message,
if anything still looks non-finite -- instead of letting you discover
it 20 minutes into training via a NaN loss.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Make sure tcir_model.py (same folder as this script) is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tcir_model import TCIRModel, MultiTaskHuberLoss  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:  # graceful fallback if tqdm isn't installed
    def tqdm(iterable, **kwargs):
        return iterable


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

class TCIRDataset(Dataset):
    """Reads image patches lazily from an HDF5 file and pairs them with
    targets / optional metadata coming from a pandas DataFrame.

    The HDF5 file handle is opened lazily (on first __getitem__ call) so
    that each DataLoader worker process gets its own independent handle
    -- h5py file objects are not safely shareable across processes.
    """

    def __init__(
        self,
        h5_path: str,
        df: pd.DataFrame,
        indices: Sequence[int],
        h5_key: str,
        target_cols: Dict[str, str],
        meta_cols: Sequence[str],
        image_mean: np.ndarray,
        image_std: np.ndarray,
        target_mean: Dict[str, float],
        target_std: Dict[str, float],
        meta_mean: Optional[np.ndarray] = None,
        meta_std: Optional[np.ndarray] = None,
    ):
        self.h5_path = h5_path
        self.h5_key = h5_key
        self.df = df.reset_index(drop=True)
        self.indices = list(indices)
        self.target_cols = target_cols
        self.meta_cols = list(meta_cols)

        self.image_mean = image_mean.reshape(-1, 1, 1).astype(np.float32)
        self.image_std = image_std.reshape(-1, 1, 1).astype(np.float32)

        self.target_mean = target_mean
        self.target_std = target_std

        self.meta_mean = meta_mean
        self.meta_std = meta_std

        self._h5: Optional[h5py.File] = None  # opened lazily per worker

    def _get_h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        h5_idx = self.indices[i]
        f = self._get_h5()

        # (201, 201, 4) -> (4, 201, 201)
        img = f[self.h5_key][h5_idx].astype(np.float32)
        img = np.transpose(img, (2, 0, 1))

        # Remember which pixels were NaN/Inf in the RAW data before we do
        # any arithmetic on them. Subtracting/dividing an inf or NaN by a
        # finite mean/std can still produce inf/NaN, so we don't rely on
        # the arithmetic to "fix" bad pixels -- we mask them out explicitly
        # afterward, using the mask computed here on the untouched data.
        invalid = ~np.isfinite(img)

        img = (img - self.image_mean) / self.image_std
        if invalid.any():
            img[invalid] = 0.0  # missing pixel -> exactly 0 in normalized space
        img_t = torch.from_numpy(img.astype(np.float32))

        row = self.df.iloc[h5_idx]

        targets = {}
        for key, col in self.target_cols.items():
            val = float(row[col])
            val = (val - self.target_mean[key]) / self.target_std[key]
            targets[key] = torch.tensor([val], dtype=torch.float32)

        if self.meta_cols:
            meta = row[self.meta_cols].to_numpy(dtype=np.float32)
            if self.meta_mean is not None and self.meta_std is not None:
                meta = (meta - self.meta_mean) / self.meta_std
            meta_t = torch.from_numpy(meta.astype(np.float32))
        else:
            meta_t = torch.zeros(0, dtype=torch.float32)

        return img_t, meta_t, targets


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    metas = torch.stack([b[1] for b in batch], dim=0) if batch[0][1].numel() > 0 else None
    targets = {
        key: torch.cat([b[2][key] for b in batch], dim=0).unsqueeze(1)
        for key in batch[0][2].keys()
    }
    return imgs, metas, targets


# --------------------------------------------------------------------------
# Stats computation (image normalization + target normalization)
# --------------------------------------------------------------------------

def compute_image_stats(
    h5_path: str, h5_key: str, indices: Sequence[int], sample_size: int = 2000, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate per-channel mean/std from a random subsample of the
    training indices (avoids loading the whole file into memory).

    Both NaN and +-Inf pixels are treated as "missing" and excluded from
    the mean/std computation. If an entire channel turns out to be
    non-finite in the sampled data, we fall back to mean=0/std=1 for that
    channel (with a warning) rather than propagating a NaN stat, which
    would otherwise NaN-out every single image at training time.
    """
    rng = np.random.RandomState(seed)
    idx = np.array(indices)
    if len(idx) > sample_size:
        idx = rng.choice(idx, size=sample_size, replace=False)
    idx = np.sort(idx)

    with h5py.File(h5_path, "r") as f:
        dset = f[h5_key]
        chunks = []
        # read in sorted chunks for faster h5py fancy indexing
        for start in range(0, len(idx), 256):
            sub = idx[start:start + 256]
            chunks.append(dset[list(sub)].astype(np.float64))  # float64 to avoid overflow in sums
        data = np.concatenate(chunks, axis=0)  # (n, 201, 201, 4)

    # Treat +-inf the same as NaN -- both mean "no valid reading here".
    # (The old version only special-cased NaN, so any +-inf sentinel in the
    # raw data would get silently replaced by a huge finite float by the
    # default np.nan_to_num behavior and blow up the mean/std.)
    data[~np.isfinite(data)] = np.nan

    with np.errstate(invalid="ignore"):
        mean = np.nanmean(data, axis=(0, 1, 2))
        std = np.nanstd(data, axis=(0, 1, 2))

    bad_mean = ~np.isfinite(mean)
    if bad_mean.any():
        print(
            f"WARNING: channel(s) {np.where(bad_mean)[0].tolist()} were entirely "
            f"NaN/Inf in the sampled training data. Falling back to mean=0 for "
            f"these channels -- consider increasing --stat-sample-size or "
            f"checking that these channels are populated for this storm subset."
        )
        mean = np.where(bad_mean, 0.0, mean)

    # NaN std (all-missing channel) and near-zero/NaN std both need a safe
    # fallback. Note: `std < 1e-6` alone does NOT catch NaN, since any
    # comparison against NaN evaluates to False in NumPy -- that was the
    # bug that let NaN stats slip through before.
    bad_std = ~np.isfinite(std) | (std < 1e-6)
    if bad_std.any():
        std = np.where(bad_std, 1.0, std)

    return mean.astype(np.float32), std.astype(np.float32)


def compute_target_stats(
    df: pd.DataFrame, target_cols: Dict[str, str]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    mean, std = {}, {}
    for key, col in target_cols.items():
        m = float(df[col].mean())
        s = float(df[col].std())
        if not np.isfinite(m):
            raise ValueError(
                f"Target column '{col}' has a non-finite mean over the training "
                f"split -- check for Inf values (NaNs should already have been "
                f"dropped by the valid-row filter in main())."
            )
        mean[key] = m
        std[key] = s if (np.isfinite(s) and s > 1e-6) else 1.0
    return mean, std


def verify_alignment(
    h5_path: str,
    h5_key: str,
    df: pd.DataFrame,
    n_total: int,
    candidate_cols: Sequence[str] = ("ID", "id", "time", "Time", "lon", "lat", "lon_grid", "lat_grid"),
    n_check: int = 30,
    seed: int = 0,
    tol: float = 1e-3,
) -> None:
    """Best-effort sanity check that CSV row i really corresponds to
    H5 row i, before we train on that assumption.

    Strategy: list every top-level dataset in the H5 file. If any of
    them share a name (case-insensitive) with a CSV column, compare
    values at a handful of sampled row indices. A mismatch means the
    "row i <-> row i" assumption is WRONG and training would silently
    pair images with the wrong labels.

    If the H5 file has no such auxiliary fields (only the raw image
    tensor), there is nothing to automatically check -- we print a
    loud warning instead of a false sense of security.
    """
    with h5py.File(h5_path, "r") as f:
        h5_keys = list(f.keys())
        print(f"H5 top-level keys found: {h5_keys}")

        # map CSV column name (lowercased) -> matching H5 key
        col_lookup = {c.lower(): c for c in df.columns}
        matches = [(k, col_lookup[k.lower()]) for k in h5_keys if k.lower() in col_lookup and k != h5_key]

        if not matches:
            print(
                "\n*** ALIGNMENT WARNING ***\n"
                "No auxiliary per-sample fields (ID/time/lat/lon-style dataset) were found "
                "in the H5 file that also appear as CSV columns, so alignment between the "
                f"'{h5_key}' image dataset and your CSV rows CANNOT be automatically verified.\n"
                "The script will proceed assuming CSV row i == H5 row i (the standard TCIR "
                "export convention), but you should confirm this yourself before trusting "
                "the trained model, e.g.:\n"
                "  - open both files and manually compare a few known storms (e.g. plot "
                "    image i and check it visually matches the lat/lon/time in CSV row i)\n"
                "  - check that len(h5['%s']) == len(csv) exactly (no silent truncation)\n"
                "  - check the CSV and H5 were generated by the same export script/run\n" % h5_key
            )
            return

        rng = np.random.RandomState(seed)
        idx = rng.choice(n_total, size=min(n_check, n_total), replace=False)
        idx = np.sort(idx)

        all_ok = True
        for h5_field, csv_col in matches:
            h5_vals = f[h5_field][list(idx)]
            csv_vals = df[csv_col].to_numpy()[idx]
            try:
                ok = np.allclose(
                    h5_vals.astype(np.float64), csv_vals.astype(np.float64),
                    atol=tol, equal_nan=True,
                )
            except (ValueError, TypeError):
                # non-numeric (e.g. byte-string IDs) -> compare as strings
                h5_str = np.array([v.decode() if isinstance(v, bytes) else str(v) for v in h5_vals])
                csv_str = csv_vals.astype(str)
                ok = bool(np.all(h5_str == csv_str))

            status = "OK" if ok else "MISMATCH"
            print(f"Alignment check on '{h5_field}' vs CSV column '{csv_col}': {status}")
            all_ok = all_ok and ok

        if not all_ok:
            raise ValueError(
                "H5/CSV alignment check FAILED. The image dataset and your label CSV do not "
                "appear to be in the same row order. Fix the alignment (e.g. sort/join both "
                "by a shared ID) before training -- proceeding would train on mismatched "
                "image/label pairs."
            )
        print("Alignment check passed on all matched fields.\n")


def compute_meta_stats(df: pd.DataFrame, meta_cols: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    if not meta_cols:
        return None, None
    mean = df[meta_cols].mean().to_numpy(dtype=np.float32)
    std = df[meta_cols].std().to_numpy(dtype=np.float32)
    bad_mean = ~np.isfinite(mean)
    if bad_mean.any():
        raise ValueError(
            f"Metadata column(s) at index {np.where(bad_mean)[0].tolist()} have a "
            f"non-finite mean over the training split -- check for Inf values."
        )
    bad_std = ~np.isfinite(std) | (std < 1e-6)
    std = np.where(bad_std, 1.0, std)
    return mean, std


def assert_finite_stats(name: str, arr) -> None:
    """Fail fast, with a clear message, if any normalization statistic is
    non-finite. Without this, a single NaN/Inf in the stats silently NaNs
    out every image/target in the dataset and you only find out much later
    from a NaN training loss."""
    if arr is None:
        return
    values = np.asarray(list(arr.values())) if isinstance(arr, dict) else np.asarray(arr)
    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"Computed normalization stats for '{name}' contain non-finite values: {values}. "
            f"This would silently NaN out training -- refusing to proceed."
        )


# --------------------------------------------------------------------------
# Train / validate loops
# --------------------------------------------------------------------------

def move_targets(targets: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in targets.items()}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    use_amp: bool,
    grad_clip: float,
) -> Dict[str, float]:
    model.train()
    running = {"vmax": 0.0, "mslp": 0.0, "r34avg": 0.0, "total": 0.0}
    n_batches = 0
    n_skipped = 0

    for imgs, metas, targets in tqdm(loader, desc="train", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        metas = metas.to(device, non_blocking=True) if metas is not None else None
        targets = move_targets(targets, device)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            preds = model(imgs, metas)
            losses = criterion(preds, targets)
            loss = losses["total"]

        if not torch.isfinite(loss):
            # Defensive guard: if a non-finite loss shows up despite the
            # input sanitization above (e.g. from an unrelated numerical
            # issue inside the model), skip this optimizer step instead of
            # corrupting the model weights with a NaN gradient update.
            n_skipped += 1
            print(f"WARNING: non-finite loss ({loss.item()}) on a training batch -- skipping this step.")
            continue

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        for k in running:
            running[k] += float(losses[k].detach().item())
        n_batches += 1

    if n_skipped:
        print(f"  ({n_skipped} batch(es) skipped this epoch due to non-finite loss)")

    return {k: v / max(n_batches, 1) for k, v in running.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    target_mean: Dict[str, float],
    target_std: Dict[str, float],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    model.eval()
    running = {"vmax": 0.0, "mslp": 0.0, "r34avg": 0.0, "total": 0.0}
    sq_err = {"vmax": 0.0, "mslp": 0.0, "r34avg": 0.0}
    n_samples = 0
    n_batches = 0

    for imgs, metas, targets in tqdm(loader, desc="val", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        metas = metas.to(device, non_blocking=True) if metas is not None else None
        targets = move_targets(targets, device)

        preds = model(imgs, metas)
        losses = criterion(preds, targets)

        bsz = imgs.shape[0]
        for k in running:
            running[k] += float(losses[k].item())
        for k in sq_err:
            # un-normalize to original physical units for an interpretable RMSE
            pred_orig = preds[k].detach().cpu().numpy() * target_std[k] + target_mean[k]
            targ_orig = targets[k].detach().cpu().numpy() * target_std[k] + target_mean[k]
            sq_err[k] += float(np.sum((pred_orig - targ_orig) ** 2))
        n_samples += bsz
        n_batches += 1

    avg_losses = {k: v / max(n_batches, 1) for k, v in running.items()}
    rmse = {k: float(np.sqrt(v / max(n_samples, 1))) for k, v in sq_err.items()}
    return avg_losses, rmse


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the TCIR intensity regression model.")

    p.add_argument("--h5", type=str, required=True, help="Path to the HDF5 file with image patches.")
    p.add_argument("--csv", type=str, required=True, help="Path to the CSV file with labels/metadata.")
    p.add_argument("--h5-key", type=str, default="matrix", help="Dataset key inside the HDF5 file.")
    p.add_argument("--output-dir", type=str, default="./runs/tcir_exp", help="Where to save checkpoints/logs.")

    p.add_argument("--vmax-col", type=str, default="Vmax", help="CSV column name for max wind speed.")
    p.add_argument("--mslp-col", type=str, default="MSLP", help="CSV column name for minimum sea-level pressure.")
    p.add_argument("--r34-col", type=str, default="R35_4qAVG", help="CSV column name for the R34-style wind-radius average.")
    p.add_argument("--meta-cols", type=str, default="", help="Comma-separated list of extra scalar metadata columns to feed the model (e.g. basin_sin,basin_cos,doy_sin,doy_cos). Leave empty to disable the metadata branch.")

    p.add_argument("--val-frac", type=float, default=0.15, help="Fraction of data held out for validation.")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--patience", type=int, default=12, help="Early-stopping patience (epochs with no val improvement).")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--amp", action="store_true", help="Use mixed-precision training (CUDA only).")
    p.add_argument("--stat-sample-size", type=int, default=4000, help="Number of training samples used to estimate image normalization stats.")

    p.add_argument(
        "--loss-weights",
        type=str,
        default="vmax:1.0,mslp:1.0,r34avg:1.0",
        help="Comma-separated key:weight pairs for the multi-task loss.",
    )

    return p.parse_args()


def parse_loss_weights(s: str) -> Dict[str, float]:
    weights = {}
    for pair in s.split(","):
        pair = pair.strip()
        if not pair:
            continue
        key, val = pair.split(":")
        weights[key.strip()] = float(val)
    return weights


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    meta_cols = [c.strip() for c in args.meta_cols.split(",") if c.strip()]
    target_cols = {"vmax": args.vmax_col, "mslp": args.mslp_col, "r34avg": args.r34_col}

    # ---------------------------------------------------------------
    # Load & align labels
    # ---------------------------------------------------------------
    print("Loading CSV labels...")
    df = pd.read_csv(args.csv)

    required_cols = list(target_cols.values()) + meta_cols
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"The following columns were not found in the CSV: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    with h5py.File(args.h5, "r") as f:
        if args.h5_key not in f:
            raise ValueError(f"Key '{args.h5_key}' not found in HDF5 file. Keys present: {list(f.keys())}")
        n_h5 = f[args.h5_key].shape[0]

    if n_h5 != len(df):
        print(
            f"WARNING: HDF5 has {n_h5} samples but CSV has {len(df)} rows. "
            f"Truncating both to the first {min(n_h5, len(df))} rows. "
            f"Verify that your CSV and HDF5 rows are actually aligned!"
        )
    n_total = min(n_h5, len(df))
    df = df.iloc[:n_total].reset_index(drop=True)

    # Best-effort check that CSV row i really is H5 row i before we train
    # on that assumption (see function docstring for what this can/can't catch).
    verify_alignment(args.h5, args.h5_key, df, n_total, seed=args.seed)

    # Drop rows with missing targets
    before = len(df)
    df = df.dropna(subset=list(target_cols.values())).reset_index(drop=True)
    # NOTE: dropping rows changes row->h5-index alignment, so we must keep
    # explicit original indices rather than relying on the new dataframe index.
    df["_h5_row"] = df.index  # placeholder, fixed just below
    # Re-derive from a fresh, un-dropped frame so h5 indices stay correct:
    df_full = pd.read_csv(args.csv).iloc[:n_total].reset_index(drop=True)
    valid_mask = df_full[list(target_cols.values())].notna().all(axis=1)
    if meta_cols:
        valid_mask &= df_full[meta_cols].notna().all(axis=1)
    valid_h5_indices = np.where(valid_mask.to_numpy())[0]
    df = df_full  # keep full frame; TCIRDataset indexes by h5 row via `indices`
    after = len(valid_h5_indices)
    print(f"Usable samples after dropping NaN targets/metadata: {after} (of {before})")

    # ---------------------------------------------------------------
    # Train / validation split
    # ---------------------------------------------------------------
    rng = np.random.RandomState(args.seed)
    shuffled = valid_h5_indices.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * args.val_frac))
    val_indices = shuffled[:n_val]
    train_indices = shuffled[n_val:]
    print(f"Train samples: {len(train_indices)} | Val samples: {len(val_indices)}")

    # ---------------------------------------------------------------
    # Normalization stats (computed on TRAIN split only)
    # ---------------------------------------------------------------
    print("Computing image normalization stats from training split...")
    image_mean, image_std = compute_image_stats(
        args.h5, args.h5_key, train_indices, sample_size=args.stat_sample_size, seed=args.seed
    )
    print(f"  image_mean={image_mean}, image_std={image_std}")

    train_df_for_stats = df.iloc[train_indices]
    target_mean, target_std = compute_target_stats(train_df_for_stats, target_cols)
    print(f"  target_mean={target_mean}")
    print(f"  target_std ={target_std}")

    meta_mean, meta_std = compute_meta_stats(train_df_for_stats, meta_cols)

    # Fail fast, with a clear message, if any stat is non-finite -- rather
    # than silently training on NaN-poisoned inputs for several epochs.
    assert_finite_stats("image_mean", image_mean)
    assert_finite_stats("image_std", image_std)
    assert_finite_stats("target_mean", target_mean)
    assert_finite_stats("target_std", target_std)
    assert_finite_stats("meta_mean", meta_mean)
    assert_finite_stats("meta_std", meta_std)

    # Persist normalization stats so they can be reused at inference time
    stats = {
        "image_mean": image_mean.tolist(),
        "image_std": image_std.tolist(),
        "target_mean": target_mean,
        "target_std": target_std,
        "meta_cols": meta_cols,
        "meta_mean": meta_mean.tolist() if meta_mean is not None else None,
        "meta_std": meta_std.tolist() if meta_std is not None else None,
        "target_cols": target_cols,
    }
    with open(os.path.join(args.output_dir, "normalization_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)

    # ---------------------------------------------------------------
    # Datasets / loaders
    # ---------------------------------------------------------------
    train_ds = TCIRDataset(
        args.h5, df, train_indices, args.h5_key, target_cols, meta_cols,
        image_mean, image_std, target_mean, target_std, meta_mean, meta_std,
    )
    val_ds = TCIRDataset(
        args.h5, df, val_indices, args.h5_key, target_cols, meta_cols,
        image_mean, image_std, target_mean, target_std, meta_mean, meta_std,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        collate_fn=collate_fn, persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        collate_fn=collate_fn, persistent_workers=args.num_workers > 0,
    )

    # ---------------------------------------------------------------
    # Model / optimizer / scheduler / loss
    # ---------------------------------------------------------------
    model = TCIRModel(in_channels=4, meta_dim=len(meta_cols)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    loss_weights = parse_loss_weights(args.loss_weights)
    criterion = MultiTaskHuberLoss(weights=loss_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ---------------------------------------------------------------
    # Training loop with early stopping on best validation total loss
    # ---------------------------------------------------------------
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    history = []

    best_ckpt_path = os.path.join(args.output_dir, "best_model.pt")
    last_ckpt_path = os.path.join(args.output_dir, "last_model.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, use_amp, args.grad_clip
        )
        val_losses, val_rmse = validate(model, val_loader, criterion, device, target_mean, target_std)
        scheduler.step()

        dt = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"train_loss={train_losses['total']:.4f} "
            f"val_loss={val_losses['total']:.4f} "
            f"| val_RMSE vmax={val_rmse['vmax']:.2f} mslp={val_rmse['mslp']:.2f} r34avg={val_rmse['r34avg']:.2f} "
            f"| lr={current_lr:.2e} | {dt:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_losses,
            "val_loss": val_losses,
            "val_rmse_original_units": val_rmse,
            "lr": current_lr,
            "time_sec": dt,
        })

        # Always keep a "last" checkpoint (handy for resuming)
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_losses["total"],
            "args": vars(args),
            "normalization_stats": stats,
        }, last_ckpt_path)

        # Save the best checkpoint by validation loss
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_losses["total"],
                "val_rmse_original_units": val_rmse,
                "args": vars(args),
                "normalization_stats": stats,
            }, best_ckpt_path)
            print(f"  -> new best model saved (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print(f"Early stopping: no improvement for {args.patience} epochs (best epoch={best_epoch}).")
            break

    with open(os.path.join(args.output_dir, "history.json"), "w") as fh:
        json.dump(history, fh, indent=2)

    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f} at epoch {best_epoch}.")
    print(f"Best model saved to: {best_ckpt_path}")
    print(f"Normalization stats saved to: {os.path.join(args.output_dir, 'normalization_stats.json')}")


if __name__ == "__main__":
    main()