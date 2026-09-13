import h5py
import numpy as np
import matplotlib.pyplot as plt

filepath = "data/testData/3DIMG_15MAY2021_2330_L1C_ASIA_MER_V01R00.h5"

channel = "TIR1"  # options: VIS, SWIR, MIR, WV, TIR1, TIR2

with h5py.File(filepath, "r") as f:
    counts = f[f"IMG_{channel}"][0].astype(np.int32)          # (1616, 1618) grey counts
    lut = f[f"IMG_{channel}_TEMP"][:] if channel != "VIS" else f["IMG_VIS_ALBEDO"][:]
    left, right = f.attrs["left_longitude"][0], f.attrs["right_longitude"][0]
    lower, upper = f.attrs["lower_latitude"][0], f.attrs["upper_latitude"][0]
    acq_time = f.attrs["Acquisition_Start_Time"]

# Apply the LUT: counts (0-1023) -> physical value
physical = lut[counts]

# Mask fill/invalid counts (commonly 1023 or negative radiance)
physical = np.where((counts <= 0) | (counts >= 1023), np.nan, physical)

plt.figure(figsize=(10, 8))
cmap = "gray" if channel == "VIS" else "inferno_r" if "TIR" in channel or channel == "WV" else "viridis"
plt.imshow(physical, cmap=cmap, extent=[left, right, lower, upper], origin="upper")
plt.colorbar(label="Albedo (%)" if channel == "VIS" else "Brightness Temp (K)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"INSAT-3D IMG_{channel} — ASIA Sector — {acq_time.decode() if isinstance(acq_time, bytes) else acq_time}")
plt.show()