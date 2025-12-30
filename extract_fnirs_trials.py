import pandas as pd
import mne
import numpy as np

# Configuration
snirf_file = '/Volumes/Extreme SSD/ds004514-download/sub-01/nirs/sub-01_task-nirs_nirs.snirf'  # Your SNIRF file
events_file = '/Volumes/Extreme SSD/cleaned nirs events files/sub1cleanedNIRS.tsv'  # Your cleaned events file
output_file = '/Volumes/Extreme SSD/cleaned NIRS trials/sub1NIRSTrials.csv'  # Output file

print("Loading SNIRF file...")
# Read the SNIRF file using MNE
raw = mne.io.read_raw_snirf(snirf_file, preload=True, verbose=False)

# Get sampling frequency
sfreq = raw.info['sfreq']
print(f"Sampling frequency: {sfreq} Hz")
print(f"Number of channels: {len(raw.ch_names)}")

# Read the cleaned events file
print("Loading events file...")
events_df = pd.read_csv(events_file, sep='\t')

# Group events into trials (every 3 consecutive rows = 1 trial)
print("Grouping events into trials...")
trials = []
for i in range(0, len(events_df), 3):
    if i + 2 < len(events_df):
        trial = {
            'trial_num': i // 3 + 1,
            'start_time': events_df.iloc[i]['onset'],
            'imagery_time': events_df.iloc[i+1]['onset'],
            'end_time': events_df.iloc[i+2]['onset'],
            'trial_type': events_df.iloc[i+1]['trial_type'],
            'stim_file': events_df.iloc[i+1]['stim_file']
        }
        trials.append(trial)

print(f"Found {len(trials)} trials")

# Extract fNIRS data for each trial
all_trial_data = []

for trial in trials:
    print(f"Extracting trial {trial['trial_num']}/{len(trials)}...", end='\r')
    
    # Convert time to samples
    start_sample = int(trial['start_time'] * sfreq)
    end_sample = int(trial['end_time'] * sfreq)
    
    # Extract data for this time window
    data, times = raw[:, start_sample:end_sample]
    
    # Create dataframe for this trial
    for time_idx, time_point in enumerate(times):
        row = {
            'trial_num': trial['trial_num'],
            'time': time_point,
            'time_from_trial_start': time_point - trial['start_time'],
            'trial_type': trial['trial_type'],
            'stim_file': trial['stim_file']
        }
        
        # Add each channel's data
        for ch_idx, ch_name in enumerate(raw.ch_names):
            row[ch_name] = data[ch_idx, time_idx]
        
        all_trial_data.append(row)

print("\nCreating final dataframe...")
df_trials = pd.DataFrame(all_trial_data)

print(f"Saving to {output_file}...")
df_trials.to_csv(output_file, index=False)

print(f"\nDone! Saved {len(df_trials)} rows across {len(trials)} trials")
print(f"Columns: {list(df_trials.columns[:10])}... (and {len(df_trials.columns)-10} more)")
print(f"File size: {df_trials.memory_usage(deep=True).sum() / 1e6:.2f} MB in memory")

# Additional fNIRS-specific info
print("\n" + "="*60)
print("fNIRS Channel Information:")
print("="*60)
print(f"Sample channel names: {raw.ch_names[:5]}")
print("\nNote: fNIRS channels typically include:")
print("  - HbO (oxygenated hemoglobin)")
print("  - HbR (deoxygenated hemoglobin)")
print("  - Channel locations indicate source-detector pairs")