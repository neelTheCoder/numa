#!/usr/bin/env python3
"""
EEG-based Animal/Tool Classification Model
==========================================
This script trains a machine learning model to predict which animal or tool
a participant is thinking of based on their EEG brain scan data.

Requirements:
- 12 CSV files (named subject_01.csv through subject_12.csv) in the same directory
- Each CSV should have columns: trial_num, time, time_from_trial_start, trial_type, 
  stim_file, A1-A32, B1-B32, EXG1-EXG8, GSR1-GSR2, Erg1-Erg2, Resp, Plet, Temp, Status
"""

import numpy as np
import pandas as pd
import glob
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def extract_label_from_filename(filename):
    """
    Extract animal/tool label from stimulus filename.
    Example: 'animals/owl_PNG26.png' -> 'owl'
             'tools/scissors_PNG28.png' -> 'scissors'
    """
    if pd.isna(filename):
        return None

    # Extract the base filename without path and extension
    basename = os.path.basename(filename)
    # Remove file extension and PNG number
    label = basename.split('_')[0].lower()
    return label


def aggregate_trial_features(trial_df, electrode_cols):
    """
    Aggregate features for a single trial by computing statistics
    across all time points within that trial.
    """
    features = []

    # Statistical features for each electrode
    for col in electrode_cols:
        if col in trial_df.columns:
            values = trial_df[col].values
            features.extend([
                np.mean(values),
                np.std(values),
                np.min(values),
                np.max(values),
                np.median(values)
            ])

    return np.array(features)


def load_and_process_subject(filepath, electrode_cols):
    """
    Load a subject's CSV file and process it into trials with aggregated features.
    Returns a DataFrame with one row per trial.
    """
    print(f"  Loading: {os.path.basename(filepath)}")

    df = pd.read_csv(filepath)

    # Extract labels from stimulus filenames
    df['label'] = df['stim_file'].apply(extract_label_from_filename)

    # Group by trial number
    trials_data = []
    trial_numbers = df['trial_num'].unique()

    for trial_num in trial_numbers:
        trial_df = df[df['trial_num'] == trial_num]

        # Get the label (should be same for all rows in trial)
        label = trial_df['label'].iloc[0]

        if label is None:
            continue

        # Aggregate features across time points in this trial
        features = aggregate_trial_features(trial_df, electrode_cols)

        trials_data.append({
            'trial_num': trial_num,
            'label': label,
            'features': features
        })

    result_df = pd.DataFrame(trials_data)
    print(f"    Processed {len(result_df)} trials")

    return result_df


def main():
    print("=" * 70)
    print("EEG-BASED ANIMAL/TOOL CLASSIFICATION")
    print("=" * 70)

    # Define electrode columns (A1-A32, B1-B32)
    electrode_cols = [f'A{i}' for i in range(1, 33)] + [f'B{i}' for i in range(1, 33)]

    # Find all CSV files
    csv_files = sorted(glob.glob('subject_*.csv'))

    if len(csv_files) == 0:
        print("\nERROR: No CSV files found matching 'subject_*.csv'")
        print("Please ensure your CSV files are named: subject_01.csv, subject_02.csv, etc.")
        return

    if len(csv_files) < 12:
        print(f"\nWARNING: Found only {len(csv_files)} CSV files. Expected 12.")
        print("Proceeding with available files...")

    print(f"\nFound {len(csv_files)} subject files")
    print(f"Using first {min(9, len(csv_files)-3)} for training, last 3 for testing\n")

    # Split into train and test files
    if len(csv_files) >= 12:
        train_files = csv_files[:9]
        test_files = csv_files[9:12]
    else:
        # If fewer than 12 files, use 70-30 split
        split_idx = int(len(csv_files) * 0.7)
        train_files = csv_files[:split_idx]
        test_files = csv_files[split_idx:]

    # Load training data
    print("STEP 1: Loading Training Data")
    print("-" * 70)
    train_data = []

    for filepath in tqdm(train_files, desc="Loading training subjects"):
        subject_df = load_and_process_subject(filepath, electrode_cols)
        train_data.append(subject_df)

    train_df = pd.concat(train_data, ignore_index=True)
    print(f"\nTotal training trials: {len(train_df)}")
    print(f"Unique labels in training: {sorted(train_df['label'].unique())}")

    # Prepare training data
    print("\nSTEP 2: Preparing Training Features")
    print("-" * 70)

    X_train = np.vstack(train_df['features'].values)
    y_train = train_df['label'].values

    print(f"Training feature matrix shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train model
    print("\nSTEP 3: Training Random Forest Classifier")
    print("-" * 70)
    print("Training in progress...")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    model.fit(X_train_scaled, y_train)

    train_accuracy = model.score(X_train_scaled, y_train)
    print(f"Training accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")

    # Test on each test subject
    print("\nSTEP 4: Testing on Hold-Out Subjects")
    print("-" * 70)

    test_results = []

    for filepath in test_files:
        subject_name = os.path.basename(filepath).replace('.csv', '')
        print(f"\nTesting on: {subject_name}")

        # Load test subject
        test_df = load_and_process_subject(filepath, electrode_cols)

        X_test = np.vstack(test_df['features'].values)
        y_test = test_df['label'].values

        # Scale test features
        X_test_scaled = scaler.transform(X_test)

        # Make predictions
        y_pred = model.predict(X_test_scaled)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        print(f"  Trials: {len(y_test)}")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

        test_results.append({
            'subject': subject_name,
            'trials': len(y_test),
            'accuracy': accuracy,
            'correct': int(accuracy * len(y_test)),
            'total': len(y_test)
        })

        # Detailed classification report
        print(f"\n  Classification Report for {subject_name}:")
        print(classification_report(y_test, y_pred, zero_division=0))

    # Summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)

    results_df = pd.DataFrame(test_results)
    print(results_df.to_string(index=False))

    avg_accuracy = np.mean([r['accuracy'] for r in test_results])
    print(f"\nAverage Test Accuracy: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
    print(f"Total Test Trials: {sum([r['total'] for r in test_results])}")
    print(f"Total Correct Predictions: {sum([r['correct'] for r in test_results])}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
