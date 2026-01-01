"""
Multimodal EEG-fNIRS Classification - OPTIMIZED & MOST EFFICIENT
=========================================================================
OPTIMIZATIONS:
- Pre-fitted estimators passed to VotingClassifier (NO double training)
- DataLoader drop_last=True to prevent BatchNorm error
- Manual estimator assignment for zero-overhead ensemble
"""

import numpy as np
import pandas as pd
import glob
import os
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm import tqdm
except ImportError:
    print("Installing PyTorch and dependencies...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'torch', 'tqdm', 'scikit-learn', 'scipy'])
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm import tqdm

from scipy.linalg import eigh
from scipy.signal import welch, butter, filtfilt, detrend, find_peaks
from scipy.stats import skew, kurtosis

# Handle trapz compatibility
try:
    from scipy.integrate import trapz
except ImportError:
    from numpy import trapz


# ============================================================================
# EEG FEATURE EXTRACTORS
# ============================================================================

class RiemannianFeatureExtractor:
    """Extract Riemannian geometry features from EEG using covariance matrices"""
    def __init__(self):
        self.reference_mean = None

    def compute_covariance(self, trial_data):
        cov = np.cov(trial_data)
        cov = cov + 1e-6 * np.eye(cov.shape[0])
        return cov

    def log_euclidean_mean(self, cov_matrices):
        log_cov = [self._logm(cov) for cov in cov_matrices]
        mean_log = np.mean(log_cov, axis=0)
        return self._expm(mean_log)

    def _logm(self, matrix):
        eigenvalues, eigenvectors = eigh(matrix)
        log_eigenvalues = np.log(np.maximum(eigenvalues, 1e-10))
        return eigenvectors @ np.diag(log_eigenvalues) @ eigenvectors.T

    def _expm(self, matrix):
        eigenvalues, eigenvectors = eigh(matrix)
        exp_eigenvalues = np.exp(eigenvalues)
        return eigenvectors @ np.diag(exp_eigenvalues) @ eigenvectors.T

    def tangent_space_projection(self, cov_matrix, reference_mean):
        ref_inv_sqrt = self._matrix_power(reference_mean, -0.5)
        transported = ref_inv_sqrt @ cov_matrix @ ref_inv_sqrt
        log_transported = self._logm(transported)

        n = log_transported.shape[0]
        indices = np.triu_indices(n)
        tangent_vector = log_transported[indices]
        tangent_vector[n:] *= np.sqrt(2)

        return tangent_vector

    def _matrix_power(self, matrix, power):
        eigenvalues, eigenvectors = eigh(matrix)
        powered_eigenvalues = np.power(np.maximum(eigenvalues, 1e-10), power)
        return eigenvectors @ np.diag(powered_eigenvalues) @ eigenvectors.T

    def fit_transform(self, trials_data):
        cov_matrices = []
        for trial in tqdm(trials_data, desc="    EEG Covariance", leave=False):
            cov_matrices.append(self.compute_covariance(trial))

        self.reference_mean = self.log_euclidean_mean(cov_matrices)

        features = []
        for cov in tqdm(cov_matrices, desc="    EEG Tangent Space", leave=False):
            features.append(self.tangent_space_projection(cov, self.reference_mean))

        return np.array(features)

    def transform(self, trials_data):
        if self.reference_mean is None:
            raise ValueError("Must fit before transform")

        cov_matrices = []
        for trial in trials_data:
            cov_matrices.append(self.compute_covariance(trial))

        features = []
        for cov in cov_matrices:
            features.append(self.tangent_space_projection(cov, self.reference_mean))

        return np.array(features)


class SpectralFeatureExtractor:
    """Extract frequency-domain features using power spectral density"""
    def __init__(self, fs=250):
        self.fs = fs
        self.freq_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }

    def extract_band_power(self, signal):
        freqs, psd = welch(signal, self.fs, nperseg=min(256, len(signal)))

        band_powers = []
        for band_name, (low, high) in self.freq_bands.items():
            idx = np.logical_and(freqs >= low, freqs <= high)
            band_power = np.trapz(psd[idx], freqs[idx])
            band_powers.append(band_power)

        return np.array(band_powers)

    def fit_transform(self, trials_data):
        all_features = []

        for trial in tqdm(trials_data, desc="    EEG Spectral", leave=False):
            trial_features = []
            for channel_data in trial:
                band_powers = self.extract_band_power(channel_data)
                trial_features.extend(band_powers)
            all_features.append(trial_features)

        return np.array(all_features)

    def transform(self, trials_data):
        all_features = []

        for trial in trials_data:
            trial_features = []
            for channel_data in trial:
                band_powers = self.extract_band_power(channel_data)
                trial_features.extend(band_powers)
            all_features.append(trial_features)

        return np.array(all_features)


# ============================================================================
# OPTIMIZED fNIRS FEATURE EXTRACTOR
# ============================================================================

class OptimizedHemodynamicFeatureExtractor:
    """OPTIMIZED fNIRS feature extraction for short trials"""
    def __init__(self, fs=7.8125):
        self.fs = fs
        self.scaler = RobustScaler()
        self.fitted = False
        self.features_per_channel = 35
        self.expected_n_pairs = None

    def minimal_preprocessing(self, data):
        data = np.asarray(data).flatten()
        if len(data) < 2:
            return data
        return detrend(data, type='linear')

    def compute_hbo_hbr(self, signal_785, signal_830):
        epsilon_hbo_785 = 0.7
        epsilon_hbr_785 = 1.5
        epsilon_hbo_830 = 2.3
        epsilon_hbr_830 = 1.0

        det = epsilon_hbo_785 * epsilon_hbr_830 - epsilon_hbo_830 * epsilon_hbr_785

        hbo = (epsilon_hbr_830 * signal_785 - epsilon_hbr_785 * signal_830) / det
        hbr = (-epsilon_hbo_830 * signal_785 + epsilon_hbo_785 * signal_830) / det

        return hbo, hbr

    def extract_amplitude_features(self, signal):
        signal = np.asarray(signal).flatten()

        if len(signal) < 2:
            return [0.0] * 12

        features = [
            np.mean(signal), np.std(signal), np.median(signal),
            np.max(signal), np.min(signal), np.ptp(signal),
            np.percentile(signal, 25), np.percentile(signal, 75),
            np.percentile(signal, 90), np.percentile(signal, 10),
            skew(signal) if len(signal) > 2 else 0.0,
            kurtosis(signal) if len(signal) > 3 else 0.0,
        ]
        return features

    def extract_shape_features(self, signal):
        signal = np.asarray(signal).flatten()

        if len(signal) < 3:
            return [0.0] * 8

        first_deriv = np.gradient(signal)
        second_deriv = np.gradient(first_deriv) if len(signal) > 2 else np.zeros_like(signal)

        features = [
            np.mean(first_deriv), np.std(first_deriv),
            np.max(np.abs(first_deriv)), np.mean(second_deriv),
            np.std(second_deriv), trapz(signal) / len(signal),
            trapz(np.abs(signal)) / len(signal), np.mean(np.abs(first_deriv)),
        ]
        return features

    def extract_spectral_features_adaptive(self, signal):
        signal = np.asarray(signal).flatten()

        if len(signal) < 15:
            return [0.0] * 4

        try:
            nperseg = min(len(signal) // 2, 32)
            if nperseg < 4:
                return [0.0] * 4

            freqs, psd = welch(signal, self.fs, nperseg=nperseg)

            vlf_idx = (freqs >= 0.01) & (freqs <= 0.1)
            vlf_power = trapz(psd[vlf_idx], freqs[vlf_idx]) if np.any(vlf_idx) else 0.0

            lf_idx = (freqs >= 0.1) & (freqs <= 0.5)
            lf_power = trapz(psd[lf_idx], freqs[lf_idx]) if np.any(lf_idx) else 0.0

            total_power = trapz(psd, freqs)
            spectral_centroid = np.sum(freqs * psd) / (np.sum(psd) + 1e-10)

            features = [vlf_power, lf_power, total_power, spectral_centroid]
        except Exception:
            features = [0.0] * 4

        return features

    def extract_peak_features(self, signal):
        signal = np.asarray(signal).flatten()

        if len(signal) < 5:
            return [0.0] * 3

        try:
            peaks, properties = find_peaks(signal, prominence=np.std(signal)*0.5)

            num_peaks = len(peaks)
            peak_prominence = np.mean(properties['prominences']) if len(peaks) > 0 else 0.0
            peak_location = peaks[np.argmax(signal[peaks])] / len(signal) if len(peaks) > 0 else 0.5

            features = [num_peaks, peak_prominence, peak_location]
        except Exception:
            features = [0.0, 0.0, 0.5]

        return features

    def extract_hemodynamic_coupling_features(self, hbo, hbr):
        if len(hbo) < 2 or len(hbr) < 2:
            return [0.0] * 8

        try:
            corr = np.corrcoef(hbo, hbr)[0, 1] if len(hbo) > 1 else 0.0
            if np.isnan(corr):
                corr = 0.0

            diff = hbo - hbr

            features = [
                corr, np.mean(diff), np.std(diff), np.max(diff),
                np.min(diff), np.mean(np.abs(diff)),
                np.sum(hbo > hbr) / len(hbo),
                np.std(hbo) / (np.std(hbr) + 1e-10),
            ]
        except Exception:
            features = [0.0] * 8

        return features

    def extract_channel_features(self, signal_785, signal_830):
        signal_785 = np.asarray(signal_785).flatten()
        signal_830 = np.asarray(signal_830).flatten()

        min_len = min(len(signal_785), len(signal_830))
        signal_785 = signal_785[:min_len]
        signal_830 = signal_830[:min_len]

        if min_len < 2:
            return [0.0] * self.features_per_channel

        filtered_785 = self.minimal_preprocessing(signal_785)
        filtered_830 = self.minimal_preprocessing(signal_830)

        hbo, hbr = self.compute_hbo_hbr(filtered_785, filtered_830)

        features = []
        features.extend(self.extract_amplitude_features(hbo))
        features.extend(self.extract_amplitude_features(hbr))
        features.extend(self.extract_shape_features(hbo))
        features.extend(self.extract_spectral_features_adaptive(hbo))
        features.extend(self.extract_peak_features(hbo))
        features.extend(self.extract_hemodynamic_coupling_features(hbo, hbr))

        features = [f if np.isfinite(f) else 0.0 for f in features]

        if len(features) < self.features_per_channel:
            features.extend([0.0] * (self.features_per_channel - len(features)))
        elif len(features) > self.features_per_channel:
            features = features[:self.features_per_channel]

        return features

    def fit_transform(self, trials_data):
        max_n_pairs = 0
        for trial in trials_data:
            n_pairs = trial.shape[0] // 2
            max_n_pairs = max(max_n_pairs, n_pairs)

        self.expected_n_pairs = max_n_pairs
        expected_features = self.expected_n_pairs * self.features_per_channel

        print(f"    Expected {self.expected_n_pairs} channel pairs, {expected_features} features per trial")

        all_features = []

        for trial in tqdm(trials_data, desc="    fNIRS Features", leave=False):
            trial_features = []
            n_pairs = trial.shape[0] // 2

            for i in range(n_pairs):
                signal_785 = trial[2*i]
                signal_830 = trial[2*i + 1]
                channel_features = self.extract_channel_features(signal_785, signal_830)
                trial_features.extend(channel_features)

            if n_pairs < self.expected_n_pairs:
                missing_features = (self.expected_n_pairs - n_pairs) * self.features_per_channel
                trial_features.extend([0.0] * missing_features)

            trial_features = trial_features[:expected_features]
            if len(trial_features) < expected_features:
                trial_features.extend([0.0] * (expected_features - len(trial_features)))

            all_features.append(trial_features)

        all_features = np.array(all_features, dtype=np.float32)
        all_features = np.nan_to_num(all_features, nan=0.0, posinf=0.0, neginf=0.0)

        print(f"    ✓ Final fNIRS features shape: {all_features.shape}")

        self.scaler.fit(all_features)
        self.fitted = True

        return self.scaler.transform(all_features)

    def transform(self, trials_data):
        if not self.fitted:
            raise ValueError("Must fit before transform")

        expected_features = self.expected_n_pairs * self.features_per_channel
        all_features = []

        for trial in trials_data:
            trial_features = []
            n_pairs = trial.shape[0] // 2

            for i in range(min(n_pairs, self.expected_n_pairs)):
                signal_785 = trial[2*i]
                signal_830 = trial[2*i + 1]
                channel_features = self.extract_channel_features(signal_785, signal_830)
                trial_features.extend(channel_features)

            if len(trial_features) < expected_features:
                trial_features.extend([0.0] * (expected_features - len(trial_features)))

            trial_features = trial_features[:expected_features]

            all_features.append(trial_features)

        all_features = np.array(all_features, dtype=np.float32)
        all_features = np.nan_to_num(all_features, nan=0.0, posinf=0.0, neginf=0.0)

        return self.scaler.transform(all_features)


# ============================================================================
# NEURAL NETWORK ARCHITECTURES
# ============================================================================

class AttentionFusion(nn.Module):
    def __init__(self, eeg_dim, fnirs_dim, fusion_dim=256):
        super(AttentionFusion, self).__init__()

        self.eeg_encoder = nn.Sequential(
            nn.Linear(eeg_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.fnirs_encoder = nn.Sequential(
            nn.Linear(fnirs_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.attention = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Tanh(),
            nn.Linear(fusion_dim, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, eeg_features, fnirs_features):
        eeg_encoded = self.eeg_encoder(eeg_features)
        fnirs_encoded = self.fnirs_encoder(fnirs_features)

        combined = torch.cat([eeg_encoded, fnirs_encoded], dim=1)
        attention_weights = self.attention(combined)

        fused = attention_weights[:, 0:1] * eeg_encoded + attention_weights[:, 1:2] * fnirs_encoded

        return fused, attention_weights


class MultimodalClassifier(nn.Module):
    def __init__(self, eeg_dim, fnirs_dim, num_classes, hidden_dim=256):
        super(MultimodalClassifier, self).__init__()

        self.eeg_dim = eeg_dim
        self.fnirs_dim = fnirs_dim

        self.fusion = AttentionFusion(eeg_dim, fnirs_dim, hidden_dim)

        self.domain_adapter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 4, num_classes)
        )

    def forward(self, eeg_features, fnirs_features):
        fused_features, attention_weights = self.fusion(eeg_features, fnirs_features)
        adapted_features = self.domain_adapter(fused_features)
        output = self.classifier(adapted_features)

        return output, attention_weights


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_eeg_data(filepath, electrode_cols):
    df = pd.read_csv(filepath)
    df['label'] = df['stim_file'].apply(
        lambda x: os.path.basename(x).split('_')[0].lower() if pd.notna(x) else None
    )
    df = df[df['label'].notna()]

    trials_data = []
    for trial_num in df['trial_num'].unique():
        trial_df = df[df['trial_num'] == trial_num]
        label = trial_df['label'].iloc[0]

        time_series = []
        for col in electrode_cols:
            if col in trial_df.columns:
                time_series.append(trial_df[col].values)

        if len(time_series) > 0:
            trials_data.append({
                'data': np.array(time_series),
                'label': label,
                'trial_num': trial_num
            })

    return trials_data


def load_fnirs_data(filepath):
    df = pd.read_csv(filepath)
    df['label'] = df['stim_file'].apply(
        lambda x: os.path.basename(x).split('_')[0].lower() if pd.notna(x) else None
    )
    df = df[df['label'].notna()]

    fnirs_cols = [col for col in df.columns if col.startswith('S') and ('785' in col or '830' in col)]
    fnirs_cols = sorted(fnirs_cols)

    trials_data = []
    for trial_num in df['trial_num'].unique():
        trial_df = df[df['trial_num'] == trial_num]
        label = trial_df['label'].iloc[0]

        time_series = []
        for col in fnirs_cols:
            if col in trial_df.columns:
                time_series.append(trial_df[col].values)

        if len(time_series) > 0:
            trials_data.append({
                'data': np.array(time_series),
                'label': label,
                'trial_num': trial_num
            })

    return trials_data


def load_subject_multimodal(subject_id, eeg_pattern='subject_{:02d}_eeg.csv',
                            fnirs_pattern='subject_{:02d}_fnirs.csv', electrode_cols=None):
    eeg_file = eeg_pattern.format(subject_id)
    fnirs_file = fnirs_pattern.format(subject_id)

    print(f"    Loading EEG: {os.path.basename(eeg_file)}")
    eeg_trials = load_eeg_data(eeg_file, electrode_cols)

    print(f"    Loading fNIRS: {os.path.basename(fnirs_file)}")
    fnirs_trials = load_fnirs_data(fnirs_file)

    eeg_dict = {trial['trial_num']: trial for trial in eeg_trials}
    fnirs_dict = {trial['trial_num']: trial for trial in fnirs_trials}

    common_trials = set(eeg_dict.keys()) & set(fnirs_dict.keys())

    multimodal_trials = []
    for trial_num in sorted(common_trials):
        if eeg_dict[trial_num]['label'] == fnirs_dict[trial_num]['label']:
            multimodal_trials.append({
                'eeg_data': eeg_dict[trial_num]['data'],
                'fnirs_data': fnirs_dict[trial_num]['data'],
                'label': eeg_dict[trial_num]['label'],
                'trial_num': trial_num
            })

    print(f"    Matched {len(multimodal_trials)} trials")
    return multimodal_trials


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def augment_minority_classes(X_eeg, X_fnirs, y, min_samples=10):
    unique_labels, counts = np.unique(y, return_counts=True)

    X_eeg_augmented = []
    X_fnirs_augmented = []
    y_augmented = []

    for label, count in tqdm(list(zip(unique_labels, counts)),
                              desc="  Augmenting classes", leave=False):
        label_indices = np.where(y == label)[0]
        eeg_samples = X_eeg[label_indices]
        fnirs_samples = X_fnirs[label_indices]

        X_eeg_augmented.append(eeg_samples)
        X_fnirs_augmented.append(fnirs_samples)
        y_augmented.extend([label] * len(eeg_samples))

        if count < min_samples:
            num_to_add = min_samples - count
            for _ in range(num_to_add):
                idx = np.random.choice(len(eeg_samples))

                eeg_base = eeg_samples[idx]
                eeg_noise = np.random.normal(0, 0.01 * np.std(eeg_base), eeg_base.shape)
                eeg_augmented = eeg_base + eeg_noise

                fnirs_base = fnirs_samples[idx]
                fnirs_noise = np.random.normal(0, 0.01 * np.std(fnirs_base), fnirs_base.shape)
                fnirs_augmented = fnirs_base + fnirs_noise

                X_eeg_augmented.append(eeg_augmented.reshape(1, -1))
                X_fnirs_augmented.append(fnirs_augmented.reshape(1, -1))
                y_augmented.append(label)

    X_eeg_augmented = np.vstack(X_eeg_augmented)
    X_fnirs_augmented = np.vstack(X_fnirs_augmented)
    y_augmented = np.array(y_augmented)

    return X_eeg_augmented, X_fnirs_augmented, y_augmented


def train_multimodal_neural_model(model, train_loader, device, num_epochs=100):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    model.train()

    epoch_pbar = tqdm(range(num_epochs), desc="Training Multimodal Neural Network",
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} epochs [{elapsed}<{remaining}]')

    for epoch in epoch_pbar:
        total_loss = 0
        correct = 0
        total = 0

        batch_pbar = tqdm(train_loader, desc=f'  Epoch {epoch+1}/{num_epochs}', leave=False)

        for eeg_data, fnirs_data, labels in batch_pbar:
            eeg_data = eeg_data.to(device)
            fnirs_data = fnirs_data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs, attention_weights = model(eeg_data, fnirs_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            batch_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })

        scheduler.step()

        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        epoch_pbar.set_postfix({
            'loss': f'{epoch_loss:.4f}',
            'acc': f'{epoch_acc:.2f}%'
        })


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("=" * 80)
    print("MULTIMODAL EEG-fNIRS CLASSIFICATION - MOST EFFICIENT VERSION")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    electrode_cols = [f'A{i}' for i in range(1, 33)] + [f'B{i}' for i in range(1, 33)]
    train_subjects = list(range(1, 11))
    test_subjects = [11, 12]

    # STEP 1: Load Training Data
    print("\n" + "=" * 80)
    print("STEP 1: Loading Multimodal Training Data (10 subjects)")
    print("=" * 80)

    train_data = []
    for subject_id in tqdm(train_subjects, desc="Loading training subjects"):
        print(f"\n  Subject {subject_id:02d}:")
        try:
            subject_trials = load_subject_multimodal(
                subject_id,
                eeg_pattern='subject_{:02d}_eeg.csv',
                fnirs_pattern='subject_{:02d}_fnirs.csv',
                electrode_cols=electrode_cols
            )
            train_data.extend(subject_trials)
        except FileNotFoundError as e:
            print(f"    WARNING: Files not found for subject {subject_id}: {e}")
            continue

    print(f"\nTotal training trials: {len(train_data)}")

    if len(train_data) == 0:
        print("\nERROR: No training data loaded.")
        return

    # STEP 2: Extract Features
    print("\n" + "=" * 80)
    print("STEP 2: Multi-Modal Feature Extraction")
    print("=" * 80)

    train_labels = [trial['label'] for trial in train_data]

    print("\n[1/2] EEG Feature Extraction")
    print("-" * 80)

    eeg_trials = [trial['eeg_data'] for trial in train_data]

    print("  Extracting Riemannian geometry features...")
    riemannian_extractor = RiemannianFeatureExtractor()
    riemannian_features = riemannian_extractor.fit_transform(eeg_trials)
    print(f"    ✓ Riemannian features shape: {riemannian_features.shape}")

    print("  Extracting spectral features...")
    spectral_extractor = SpectralFeatureExtractor()
    spectral_features = spectral_extractor.fit_transform(eeg_trials)
    print(f"    ✓ Spectral features shape: {spectral_features.shape}")

    eeg_features = np.hstack([riemannian_features, spectral_features])
    print(f"  ✓ Combined EEG features shape: {eeg_features.shape}")

    print("\n[2/2] fNIRS Feature Extraction")
    print("-" * 80)

    fnirs_trials = [trial['fnirs_data'] for trial in train_data]

    print("  Extracting optimized hemodynamic features...")
    hemodynamic_extractor = OptimizedHemodynamicFeatureExtractor()
    fnirs_features = hemodynamic_extractor.fit_transform(fnirs_trials)

    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_labels)
    num_classes = len(label_encoder.classes_)
    print(f"\nNumber of classes: {num_classes}")
    print(f"Classes: {list(label_encoder.classes_)}")

    print("\nAugmenting minority classes...")
    eeg_augmented, fnirs_augmented, y_augmented = augment_minority_classes(
        eeg_features, fnirs_features, train_labels_encoded, min_samples=15
    )
    print(f"  ✓ Augmented EEG shape: {eeg_augmented.shape}")
    print(f"  ✓ Augmented fNIRS shape: {fnirs_augmented.shape}")

    print("\nNormalizing features...")
    eeg_scaler = StandardScaler()
    fnirs_scaler = StandardScaler()

    eeg_normalized = eeg_scaler.fit_transform(eeg_augmented)
    fnirs_normalized = fnirs_scaler.fit_transform(fnirs_augmented)
    print(f"  ✓ Normalized EEG shape: {eeg_normalized.shape}")
    print(f"  ✓ Normalized fNIRS shape: {fnirs_normalized.shape}")

    # STEP 3: Train Ensemble Models - MOST EFFICIENT (NO DOUBLE TRAINING)
    print("\n" + "=" * 80)
    print("STEP 3: Training Modality-Specific Ensemble Models")
    print("=" * 80)

    # EEG Ensemble - OPTIMIZED
    print("\n[1/2] Training EEG Ensemble")
    print("-" * 80)

    eeg_estimators_config = [
        ('gb', GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                         max_depth=5, random_state=42, verbose=1)),
        ('svc', SVC(kernel='rbf', C=10, gamma='scale', probability=True,
                   random_state=42, verbose=False)),
        ('lda', LinearDiscriminantAnalysis()),
        ('knn', KNeighborsClassifier(n_neighbors=7, weights='distance')),
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42, verbose=1, n_jobs=-1))
    ]

    eeg_fitted_estimators = []

    for idx, (name, estimator) in enumerate(eeg_estimators_config, 1):
        print(f"\n  [{idx}/5] Training {name.upper()}...")
        estimator.fit(eeg_normalized, y_augmented)
        train_acc = estimator.score(eeg_normalized, y_augmented)
        print(f"    ✓ {name.upper()} complete (acc: {train_acc*100:.2f}%)")
        eeg_fitted_estimators.append((name, estimator))

    # EFFICIENT FIX: Manually assign pre-fitted estimators (NO retraining)
    print("\n  Assembling EEG VotingClassifier (no retraining)...")
    eeg_ensemble = VotingClassifier(estimators=eeg_fitted_estimators, voting='soft')

    # Manually set the already-fitted estimators to avoid retraining
    eeg_ensemble.estimators_ = [est for name, est in eeg_fitted_estimators]
    eeg_ensemble.le_ = LabelEncoder().fit(y_augmented)
    eeg_ensemble.classes_ = eeg_ensemble.le_.classes_
    eeg_ensemble.named_estimators_ = dict(eeg_fitted_estimators)

    eeg_train_acc = eeg_ensemble.score(eeg_normalized, y_augmented)
    print(f"  ✓ EEG Ensemble training accuracy: {eeg_train_acc:.4f} ({eeg_train_acc*100:.2f}%)")

    # fNIRS Ensemble - OPTIMIZED
    print("\n[2/2] Training fNIRS Ensemble")
    print("-" * 80)

    fnirs_estimators_config = [
        ('gb', GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                         max_depth=5, random_state=42, verbose=1)),
        ('svc', SVC(kernel='rbf', C=10, gamma='scale', probability=True,
                   random_state=42, verbose=False)),
        ('lda', LinearDiscriminantAnalysis()),
        ('knn', KNeighborsClassifier(n_neighbors=7, weights='distance')),
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42, verbose=1, n_jobs=-1))
    ]

    fnirs_fitted_estimators = []

    for idx, (name, estimator) in enumerate(fnirs_estimators_config, 1):
        print(f"\n  [{idx}/5] Training {name.upper()}...")
        estimator.fit(fnirs_normalized, y_augmented)
        train_acc = estimator.score(fnirs_normalized, y_augmented)
        print(f"    ✓ {name.upper()} complete (acc: {train_acc*100:.2f}%)")
        fnirs_fitted_estimators.append((name, estimator))

    # EFFICIENT FIX: Manually assign pre-fitted estimators (NO retraining)
    print("\n  Assembling fNIRS VotingClassifier (no retraining)...")
    fnirs_ensemble = VotingClassifier(estimators=fnirs_fitted_estimators, voting='soft')

    # Manually set the already-fitted estimators to avoid retraining
    fnirs_ensemble.estimators_ = [est for name, est in fnirs_fitted_estimators]
    fnirs_ensemble.le_ = LabelEncoder().fit(y_augmented)
    fnirs_ensemble.classes_ = fnirs_ensemble.le_.classes_
    fnirs_ensemble.named_estimators_ = dict(fnirs_fitted_estimators)

    fnirs_train_acc = fnirs_ensemble.score(fnirs_normalized, y_augmented)
    print(f"  ✓ fNIRS Ensemble training accuracy: {fnirs_train_acc:.4f} ({fnirs_train_acc*100:.2f}%)")

    # STEP 4: Train Multimodal Neural Network - FIXED WITH drop_last=True
    print("\n" + "=" * 80)
    print("STEP 4: Training Multimodal Neural Network")
    print("=" * 80)

    eeg_dim = eeg_normalized.shape[1]
    fnirs_dim = fnirs_normalized.shape[1]

    multimodal_model = MultimodalClassifier(eeg_dim, fnirs_dim, num_classes, hidden_dim=256).to(device)

    total_params = sum(p.numel() for p in multimodal_model.parameters())
    print(f"\n  Multimodal network parameters: {total_params:,}")

    eeg_tensor = torch.FloatTensor(eeg_normalized)
    fnirs_tensor = torch.FloatTensor(fnirs_normalized)
    labels_tensor = torch.LongTensor(y_augmented)
    train_dataset = TensorDataset(eeg_tensor, fnirs_tensor, labels_tensor)

    # FIX: Add drop_last=True to prevent BatchNorm error with single-sample batches
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)

    print(f"  Training batches: {len(train_loader)}")

    train_multimodal_neural_model(multimodal_model, train_loader, device, num_epochs=100)

    # STEP 5: Test on Hold-Out Subjects
    print("\n" + "=" * 80)
    print("STEP 5: Testing on Hold-Out Subjects")
    print("=" * 80)

    test_results = []

    for subject_id in tqdm(test_subjects, desc="\nTesting subjects"):
        print(f"\n{'='*80}")
        print(f"Testing on Subject {subject_id:02d}")
        print(f"{'='*80}")

        try:
            test_data = load_subject_multimodal(
                subject_id,
                eeg_pattern='subject_{:02d}_eeg.csv',
                fnirs_pattern='subject_{:02d}_fnirs.csv',
                electrode_cols=electrode_cols
            )
        except FileNotFoundError as e:
            print(f"  WARNING: Files not found for subject {subject_id}: {e}")
            continue

        if len(test_data) == 0:
            continue

        test_eeg_trials = [trial['eeg_data'] for trial in test_data]
        test_fnirs_trials = [trial['fnirs_data'] for trial in test_data]
        test_labels = [trial['label'] for trial in test_data]

        print("\n  Extracting test features...")
        test_eeg_riem = riemannian_extractor.transform(test_eeg_trials)
        test_eeg_spec = spectral_extractor.transform(test_eeg_trials)
        test_eeg_combined = np.hstack([test_eeg_riem, test_eeg_spec])
        test_eeg_normalized = eeg_scaler.transform(test_eeg_combined)

        test_fnirs_features = hemodynamic_extractor.transform(test_fnirs_trials)
        test_fnirs_normalized = fnirs_scaler.transform(test_fnirs_features)

        test_labels_encoded = []
        for label in test_labels:
            if label in label_encoder.classes_:
                test_labels_encoded.append(label_encoder.transform([label])[0])
            else:
                test_labels_encoded.append(-1)
        test_labels_encoded = np.array(test_labels_encoded)

        print("\n  Making predictions...")
        eeg_pred = eeg_ensemble.predict(test_eeg_normalized)
        eeg_acc = accuracy_score(test_labels_encoded, eeg_pred)

        fnirs_pred = fnirs_ensemble.predict(test_fnirs_normalized)
        fnirs_acc = accuracy_score(test_labels_encoded, fnirs_pred)

        multimodal_model.eval()
        with torch.no_grad():
            test_eeg_tensor = torch.FloatTensor(test_eeg_normalized).to(device)
            test_fnirs_tensor = torch.FloatTensor(test_fnirs_normalized).to(device)
            outputs, attention = multimodal_model(test_eeg_tensor, test_fnirs_tensor)
            _, multimodal_pred = torch.max(outputs, 1)
            multimodal_pred = multimodal_pred.cpu().numpy()

        multimodal_acc = accuracy_score(test_labels_encoded, multimodal_pred)

        final_pred = []
        for i in range(len(eeg_pred)):
            votes = [eeg_pred[i], fnirs_pred[i], multimodal_pred[i]]
            final_pred.append(max(set(votes), key=votes.count))
        final_pred = np.array(final_pred)

        final_acc = accuracy_score(test_labels_encoded, final_pred)

        print(f"\n  Results for Subject {subject_id:02d}:")
        print(f"  {'-'*76}")
        print(f"    Trials:             {len(test_labels_encoded)}")
        print(f"    EEG Ensemble:       {eeg_acc:.4f} ({eeg_acc*100:.2f}%)")
        print(f"    fNIRS Ensemble:     {fnirs_acc:.4f} ({fnirs_acc*100:.2f}%)")
        print(f"    Multimodal NN:      {multimodal_acc:.4f} ({multimodal_acc*100:.2f}%)")
        print(f"    Final Ensemble:     {final_acc:.4f} ({final_acc*100:.2f}%)")

        # Determine best model
        best_acc = max(eeg_acc, fnirs_acc, multimodal_acc, final_acc)
        if best_acc == final_acc:
            best_pred = final_pred
            best_name = "Final Ensemble"
        elif best_acc == multimodal_acc:
            best_pred = multimodal_pred
            best_name = "Multimodal NN"
        elif best_acc == eeg_acc:
            best_pred = eeg_pred
            best_name = "EEG Ensemble"
        else:
            best_pred = fnirs_pred
            best_name = "fNIRS Ensemble"

        print(f"\n  Best Model: {best_name} ({best_acc*100:.2f}%)")

        test_results.append({
            'subject': f'Subject_{subject_id:02d}',
            'trials': len(test_labels_encoded),
            'eeg_acc': eeg_acc,
            'fnirs_acc': fnirs_acc,
            'multimodal_acc': multimodal_acc,
            'final_acc': final_acc,
            'best_acc': best_acc,
            'best_model': best_name
        })

    # STEP 6: Summary and Save
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)

    if len(test_results) > 0:
        results_df = pd.DataFrame(test_results)
        print("\n" + results_df.to_string(index=False))

        print(f"\n{'-'*80}")
        print("AVERAGE ACCURACIES:")
        print(f"{'-'*80}")
        print(f"  EEG Ensemble:       {np.mean([r['eeg_acc'] for r in test_results])*100:.2f}%")
        print(f"  fNIRS Ensemble:     {np.mean([r['fnirs_acc'] for r in test_results])*100:.2f}%")
        print(f"  Multimodal NN:      {np.mean([r['multimodal_acc'] for r in test_results])*100:.2f}%")
        print(f"  Final Ensemble:     {np.mean([r['final_acc'] for r in test_results])*100:.2f}%")
        print(f"  Best Model Avg:     {np.mean([r['best_acc'] for r in test_results])*100:.2f}%")

    # Save models
    print("\n" + "=" * 80)
    print("SAVING MODELS")
    print("=" * 80)

    save_dir = 'saved_multimodal_models'
    os.makedirs(save_dir, exist_ok=True)

    multimodal_path = os.path.join(save_dir, 'multimodal_complete.pth')
    torch.save({
        'multimodal_model_state': multimodal_model.state_dict(),
        'model_architecture': {
            'eeg_dim': eeg_dim,
            'fnirs_dim': fnirs_dim,
            'num_classes': num_classes,
            'hidden_dim': 256
        },
        'riemannian_extractor': riemannian_extractor,
        'spectral_extractor': spectral_extractor,
        'hemodynamic_extractor': hemodynamic_extractor,
        'eeg_scaler': eeg_scaler,
        'fnirs_scaler': fnirs_scaler,
        'label_encoder': label_encoder,
        'eeg_ensemble': eeg_ensemble,
        'fnirs_ensemble': fnirs_ensemble,
        'classes': list(label_encoder.classes_),
        'num_classes': num_classes,
        'train_subjects': train_subjects,
        'test_subjects': test_subjects,
        'test_results': test_results
    }, multimodal_path)
    print(f"  ✓ Complete model saved to: {multimodal_path}")

    # Save metadata
    metadata_path = os.path.join(save_dir, 'model_metadata.txt')
    with open(metadata_path, 'w') as f:
        f.write("Multimodal EEG-fNIRS Classification Model - MOST EFFICIENT VERSION\n")
        f.write("="*80 + "\n\n")
        f.write(f"Training Date: {pd.Timestamp.now()}\n\n")
        f.write(f"Multimodal Neural Network Architecture:\n")
        f.write(f"  - EEG input dim: {eeg_dim}\n")
        f.write(f"  - fNIRS input dim: {fnirs_dim}\n")
        f.write(f"  - Hidden dim: 256\n")
        f.write(f"  - Output classes: {num_classes}\n")
        f.write(f"  - Total parameters: {total_params:,}\n\n")
        f.write(f"Label Classes ({num_classes}):\n")
        for i, cls in enumerate(label_encoder.classes_, 1):
            f.write(f"  {i}. {cls}\n")
        f.write(f"\nTraining Subjects: {train_subjects}\n")
        f.write(f"Test Subjects: {test_subjects}\n")
        f.write(f"Total Training Trials: {len(train_data)}\n\n")

        if test_results:
            f.write("="*80 + "\n")
            f.write("TEST RESULTS\n")
            f.write("="*80 + "\n\n")
            for result in test_results:
                f.write(f"{result['subject']} ({result['trials']} trials):\n")
                f.write(f"  EEG Ensemble:   {result['eeg_acc']*100:.2f}%\n")
                f.write(f"  fNIRS Ensemble: {result['fnirs_acc']*100:.2f}%\n")
                f.write(f"  Multimodal NN:  {result['multimodal_acc']*100:.2f}%\n")
                f.write(f"  Final Ensemble: {result['final_acc']*100:.2f}%\n")
                f.write(f"  Best: {result['best_model']} ({result['best_acc']*100:.2f}%)\n\n")

            f.write("="*80 + "\n")
            f.write("AVERAGE ACCURACIES\n")
            f.write("="*80 + "\n")
            f.write(f"EEG Ensemble:   {np.mean([r['eeg_acc'] for r in test_results])*100:.2f}%\n")
            f.write(f"fNIRS Ensemble: {np.mean([r['fnirs_acc'] for r in test_results])*100:.2f}%\n")
            f.write(f"Multimodal NN:  {np.mean([r['multimodal_acc'] for r in test_results])*100:.2f}%\n")
            f.write(f"Final Ensemble: {np.mean([r['final_acc'] for r in test_results])*100:.2f}%\n")
            f.write(f"Best Model Avg: {np.mean([r['best_acc'] for r in test_results])*100:.2f}%\n")

    print(f"  ✓ Metadata saved to: {metadata_path}")

    print(f"\n✓ All models successfully saved to '{save_dir}/' directory")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
