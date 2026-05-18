#!/usr/bin/env python3
"""
Segment EEG BDF file into per-stimulus epochs.

Run from the folder containing:
  sub-01_task-eeg_events.txt   (TSV events file)
  sub-01_task-eeg_eeg.bdf      (raw EEG recording)

Output structure:
  stimuli/
    owl/instance1.bdf  ...
    scissors/instance1.bdf  ...

Dependencies:
    pip install mne pyEDFlib
"""

import os, re, sys, csv
from pathlib import Path

try:
    import mne
except ImportError:
    sys.exit("pip install mne")

try:
    import pyedflib
except ImportError:
    sys.exit("pip install pyEDFlib")

import numpy as np

EVENTS_FILE = "sub-01_task-eeg_events.txt"
BDF_FILE    = "sub-01_task-eeg_eeg.bdf"
OUTPUT_DIR  = Path("stimuli")
TRIAL_TYPES = {"animal_image", "tool_image"}


def stim_to_label(stim_file: str) -> str:
    basename = os.path.splitext(os.path.basename(stim_file))[0]
    label = re.split(r"_PNG|_\d", basename)[0]
    return re.sub(r"[^a-zA-Z0-9_-]", "_", label).lower()


def write_bdf(path: Path, raw: mne.io.BaseRaw):
    """Write a cropped Raw object to BDF using pyEDFlib."""
    data, times = raw[:]          # shape (n_ch, n_samples), volts
    n_ch, n_samp = data.shape
    sfreq = raw.info["sfreq"]
    ch_names = raw.ch_names

    f = pyedflib.EdfWriter(str(path), n_ch, file_type=pyedflib.FILETYPE_BDF)
    f.setDatarecordDuration(1)    # 1-second records

    headers = []
    for i, ch in enumerate(ch_names):
        ch_data = data[i]
        pmin = float(np.min(ch_data))
        pmax = float(np.max(ch_data))
        # Avoid zero-range channels
        if pmin == pmax:
            pmax = pmin + 1e-6
        # BDF digital range is 24-bit signed: -8388608 .. 8388607
        headers.append({
            "label":            ch[:16],
            "dimension":        "uV",
            "sample_frequency": int(sfreq),
            "physical_min":     pmin,
            "physical_max":     pmax,
            "digital_min":      -8388608,
            "digital_max":      8388607,
            "transducer":       "",
            "prefilter":        "",
        })

    f.setSignalHeaders(headers)
    # Write channel by channel
    f.writeSamples([data[i] for i in range(n_ch)])
    f.close()


# ── Parse events ───────────────────────────────────────────────────────────────
print(f"Reading events from {EVENTS_FILE} ...")
stimuli_events = []

with open(EVENTS_FILE, newline="") as fh:
    reader = csv.DictReader(fh, delimiter="\t")
    rows = list(reader)

for i, row in enumerate(rows):
    if row["trial_type"] in TRIAL_TYPES and row["stim_file"] not in ("n/a", ""):
        onset = float(row["onset"])
        label = stim_to_label(row["stim_file"])
        end   = None
        for j in range(i + 1, len(rows)):
            if rows[j]["trial_type"] in ("fixation_cross", "animal_image", "tool_image"):
                end = float(rows[j]["onset"])
                break
        stimuli_events.append((onset, end, label))

print(f"Found {len(stimuli_events)} stimulus events.")
print(f"Unique stimuli ({len(set(l for _,_,l in stimuli_events))}): "
      f"{sorted(set(l for _,_,l in stimuli_events))}")

# ── Load BDF ──────────────────────────────────────────────────────────────────
print(f"\nLoading {BDF_FILE} ...")
raw = mne.io.read_raw_bdf(BDF_FILE, preload=True, verbose=False)
total_duration = raw.times[-1]
print(f"  Sampling rate : {raw.info['sfreq']} Hz")
print(f"  Total duration: {total_duration:.2f} s")

# ── Slice and save ────────────────────────────────────────────────────────────
instance_counter: dict[str, int] = {}
failed = []

for onset, end, label in stimuli_events:
    idx = instance_counter.get(label, 0) + 1
    instance_counter[label] = idx

    t_start = onset
    t_end   = end if end is not None else min(onset + 17.0, total_duration)
    t_end   = min(t_end, total_duration)

    if t_start >= total_duration:
        print(f"  SKIP {label}/instance{idx}: onset beyond file")
        continue

    epoch = raw.copy().crop(tmin=t_start, tmax=t_end)

    out_dir = OUTPUT_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"instance{idx}.bdf"

    try:
        write_bdf(out_file, epoch)
        print(f"  Saved {out_file}  [{t_start:.3f} – {t_end:.3f} s]")
    except Exception as e:
        print(f"  ERROR {out_file}: {e}")
        failed.append(str(out_file))

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n── Summary ──────────────────────────────────────────────────────")
total_saved = sum(instance_counter.values()) - len(failed)
print(f"Total files saved : {total_saved}")
print(f"Unique stimuli    : {len(instance_counter)}")
for lbl in sorted(instance_counter):
    print(f"  {lbl:35s}  {instance_counter[lbl]} instance(s)")
if failed:
    print(f"\nFailed ({len(failed)}):")
    for f in failed:
        print(f"  {f}")
print("Done.")