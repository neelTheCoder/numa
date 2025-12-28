import os
import math
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ----------------------------
# 1. Load BDF file as Raw
# ----------------------------
print("Step 1/4: Loading BDF file...")
bdf_path = "Subject 1/sub-01_task-eeg_eeg.bdf"  # change to your file
raw = mne.io.read_raw_bdf(bdf_path, preload=True)  # if this alone kills RAM, use preload='memmap' [web:21]
print("  -> BDF loaded.")
print(raw)
print(raw.info)

ch_names = raw.ch_names
sfreq = raw.info["sfreq"]
dt_orig = 1.0 / sfreq
print(f"Original sfreq: {sfreq} Hz (dt = {dt_orig} s)")

# ----------------------------
# 2. Set up 0.1 s sampling grid
# ----------------------------
print("\nStep 2/4: Setting up 0.1 s sampling grid...")
t_step = 0.1  # seconds
t_min = 0.0
t_max = raw.times[-1]  # duration of recording [web:47]
n_times_out = int(math.floor((t_max - t_min) / t_step)) + 1
times_out = t_min + np.arange(n_times_out) * t_step
print(f"  -> Duration: {t_max:.2f} s")
print(f"  -> Output points (0.1 s): {n_times_out}")

# ----------------------------
# 3. Export to CSV in time chunks (low RAM)
# ----------------------------
print("\nStep 3/4: Exporting downsampled data (0.1 s step) to CSV by chunks...")
csv_path = os.path.splitext(bdf_path)[0] + "_eeg_0p1s.csv"
chunk_points = 10_000  # number of 0.1 s points per chunk (1000 s); reduce if still heavy

n_chunks = math.ceil(n_times_out / chunk_points)
print(f"  -> Will write {n_chunks} chunks of up to {chunk_points} time points each.")

with open(csv_path, "w", newline="") as f:
    header_written = False

    for ci in range(n_chunks):
        start_idx = ci * chunk_points
        stop_idx = min((ci + 1) * chunk_points, n_times_out)
        if start_idx >= stop_idx:
            break

        t_chunk = times_out[start_idx:stop_idx]

        # map desired times to nearest sample indices in the original data [web:47]
        sample_idx = raw.time_as_index(t_chunk)

        # fetch only those samples; this uses raw.get_data slicing instead of loading/resampling all at once [web:47][web:104]
        data_chunk = raw.get_data(start=sample_idx[0], stop=sample_idx[-1] + 1)

        # select the exact columns corresponding to the chosen sample indices
        # (time_as_index may not be perfectly regular, so we index explicitly)
        # data_chunk is (n_channels, n_samples_window); we map indices within window
        local_idx = sample_idx - sample_idx[0]
        data_chunk_sel = data_chunk[:, local_idx]  # (n_channels, len(t_chunk))

        # transpose to (n_times, n_channels)
        data_chunk_T = data_chunk_sel.T
        df_chunk = pd.DataFrame(data_chunk_T, columns=ch_names)
        df_chunk.insert(0, "time_s", t_chunk)

        # write header only once
        df_chunk.to_csv(f, index=False, header=not header_written)
        header_written = True

        pct = (stop_idx / n_times_out) * 100
        print(f"  -> Chunk {ci+1}/{n_chunks}: "
              f"{start_idx}-{stop_idx} / {n_times_out} time points ({pct:.1f}%)")

print(f"  -> Finished writing 0.1 s CSV: {csv_path}")

# ----------------------------
# 4. Plot a few channels (optional, original resolution)
# ----------------------------
print("\nStep 4/4: Plotting a few channels at original resolution...")

channels_to_plot = ["Fz", "Cz", "Pz"]  # adapt to your montage
name_to_idx = {name: idx for idx, name in enumerate(ch_names)}

t_start = 0.0
t_stop = min(10.0, raw.times[-1])  # first 10 s
idx_start = int(t_start * sfreq)
idx_stop = int(t_stop * sfreq)

data, times = raw.get_data(return_times=True)  # beware: full load; if RAM is tight, slice get_data with start/stop [web:44]
plt.figure(figsize=(12, 6))
offset = 0.0
offset_step = 5e-5

for ch_name in channels_to_plot:
    if ch_name not in name_to_idx:
        print(f"  -> Channel {ch_name} not found, skipping.")
        continue
    ch_idx = name_to_idx[ch_name]
    y = data[ch_idx, idx_start:idx_stop]
    x = times[idx_start:idx_stop]
    plt.plot(x, y + offset, label=ch_name)
    offset += offset_step

plt.xlabel("Time (s)")
plt.ylabel("EEG (V, vertically offset per channel)")
plt.title("EEG channels over time (Matplotlib)")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()
