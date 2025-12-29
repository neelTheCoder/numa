#Takes csv of events and outputs csv with only required rows

import pandas as pd

# Read the TSV file
df = pd.read_csv('/Volumes/Extreme SSD/ds004514-download/sub-09/eeg/sub-09_task-eeg_events.tsv', sep='\t')

# Find indices of visual_imagery_task rows
visual_imagery_indices = df[df['trial_type'].str.contains('visual_imagery_task', na=False)].index

# Collect rows to keep: visual imagery tasks + blank screens before and after
rows_to_keep = []
for idx in visual_imagery_indices:
    # Look backwards to find the image file (animal_image or tool_image)
    stim_file = None
    for i in range(1, 6):
        if idx - i >= 0:
            trial_type = df.loc[idx-i, 'trial_type']
            if trial_type in ['animal_image', 'tool_image']:
                # Found the image, get its stim_file
                stim_file = df.loc[idx-i, 'stim_file']
                break
    
    # Copy the stim_file to the visual imagery task row
    if stim_file is not None and pd.notna(stim_file):
        df.loc[idx, 'stim_file'] = stim_file
    
    # Add the blank_white_screen before (if exists and is blank_white_screen)
    if idx > 0 and df.loc[idx-1, 'trial_type'] == 'blank_white_screen':
        rows_to_keep.append(idx-1)

    # Add the blank_white_screen after (if exists and is blank_white_screen)
    if idx < len(df)-1 and df.loc[idx+1, 'trial_type'] == 'blank_white_screen':
        rows_to_keep.append(idx+1)
    
    # Add the visual imagery task itself
    rows_to_keep.append(idx)

# Remove duplicates and sort
rows_to_keep = sorted(set(rows_to_keep))

# Create filtered dataframe
filtered_df = df.loc[rows_to_keep]

# Save to new TSV file
filtered_df.to_csv('/Volumes/Extreme SSD/sub9cleaned.tsv', sep='\t', index=False)

print(f"Original file had {len(df)} rows")
print(f"Filtered file has {len(filtered_df)} rows")
print(f"Found {len(visual_imagery_indices)} visual imagery tasks")
print(f"\nRows per visual imagery task (including blanks): {len(filtered_df) / len(visual_imagery_indices):.1f} on average")
print("\nFirst few rows of filtered data:")
print(filtered_df.head(15))
print("\nSample visual imagery tasks with stim_files:")
print(filtered_df[filtered_df['trial_type'].str.contains('visual_imagery_task', na=False)][['trial_type', 'stim_file']].head(10))
