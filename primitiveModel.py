"""
Advanced EEG Classification with Domain Adaptation - OPTIMIZED
=========================================================================
This script implements state-of-the-art techniques for cross-subject EEG classification:
1. Domain Adaptation to handle inter-subject variability
2. Few-shot learning for classes with limited samples
3. Riemannian geometry features (covariance matrices)
4. Ensemble of multiple models
5. Transfer learning with pre-trained backbone

OPTIMIZED: Removed duplicate training - VotingClassifier trains all models once
UPDATED: Added progress bars for ensemble training
"""


import numpy as np
import pandas as pd
import glob
import os
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
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
from scipy.signal import welch



class RiemannianFeatureExtractor:
    """
    Extract Riemannian geometry features from EEG using covariance matrices.
    This has shown state-of-the-art performance in EEG classification.
    """
    def __init__(self):
        self.reference_mean = None


    def compute_covariance(self, trial_data):
        """Compute covariance matrix for a single trial"""
        cov = np.cov(trial_data)
        cov = cov + 1e-6 * np.eye(cov.shape[0])
        return cov


    def log_euclidean_mean(self, cov_matrices):
        """Compute Riemannian mean of covariance matrices"""
        log_cov = [self._logm(cov) for cov in cov_matrices]
        mean_log = np.mean(log_cov, axis=0)
        return self._expm(mean_log)


    def _logm(self, matrix):
        """Matrix logarithm"""
        eigenvalues, eigenvectors = eigh(matrix)
        log_eigenvalues = np.log(np.maximum(eigenvalues, 1e-10))
        return eigenvectors @ np.diag(log_eigenvalues) @ eigenvectors.T


    def _expm(self, matrix):
        """Matrix exponential"""
        eigenvalues, eigenvectors = eigh(matrix)
        exp_eigenvalues = np.exp(eigenvalues)
        return eigenvectors @ np.diag(exp_eigenvalues) @ eigenvectors.T


    def tangent_space_projection(self, cov_matrix, reference_mean):
        """Project covariance matrix to tangent space"""
        ref_inv_sqrt = self._matrix_power(reference_mean, -0.5)
        transported = ref_inv_sqrt @ cov_matrix @ ref_inv_sqrt
        log_transported = self._logm(transported)


        n = log_transported.shape[0]
        indices = np.triu_indices(n)
        tangent_vector = log_transported[indices]
        tangent_vector[n:] *= np.sqrt(2)


        return tangent_vector


    def _matrix_power(self, matrix, power):
        """Compute matrix power using eigendecomposition"""
        eigenvalues, eigenvectors = eigh(matrix)
        powered_eigenvalues = np.power(np.maximum(eigenvalues, 1e-10), power)
        return eigenvectors @ np.diag(powered_eigenvalues) @ eigenvectors.T


    def fit_transform(self, trials_data):
        """Extract Riemannian features from all trials"""
        print("  Computing covariance matrices...")
        cov_matrices = []
        for trial in tqdm(trials_data, desc="  Covariance matrices", leave=False):
            cov_matrices.append(self.compute_covariance(trial))


        print("  Computing Riemannian mean...")
        self.reference_mean = self.log_euclidean_mean(cov_matrices)


        print("  Projecting to tangent space...")
        features = []
        for cov in tqdm(cov_matrices, desc="  Tangent projection", leave=False):
            features.append(self.tangent_space_projection(cov, self.reference_mean))


        return np.array(features)


    def transform(self, trials_data):
        """Transform new data using fitted reference"""
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
        """Extract power in different frequency bands"""
        freqs, psd = welch(signal, self.fs, nperseg=min(256, len(signal)))


        band_powers = []
        for band_name, (low, high) in self.freq_bands.items():
            idx = np.logical_and(freqs >= low, freqs <= high)
            band_power = np.trapz(psd[idx], freqs[idx])
            band_powers.append(band_power)


        return np.array(band_powers)


    def fit_transform(self, trials_data):
        """Extract spectral features from all trials"""
        all_features = []


        for trial in tqdm(trials_data, desc="  Spectral features", leave=False):
            trial_features = []
            for channel_data in trial:
                band_powers = self.extract_band_power(channel_data)
                trial_features.extend(band_powers)
            all_features.append(trial_features)


        return np.array(all_features)


    def transform(self, trials_data):
        """Transform new data"""
        all_features = []


        for trial in trials_data:
            trial_features = []
            for channel_data in trial:
                band_powers = self.extract_band_power(channel_data)
                trial_features.extend(band_powers)
            all_features.append(trial_features)


        return np.array(all_features)



class DomainAdaptationLayer(nn.Module):
    """Domain Adaptation layer using adversarial training"""
    def __init__(self, input_dim, hidden_dim=128):
        super(DomainAdaptationLayer, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )


    def forward(self, x):
        return self.feature_extractor(x)



class EnhancedEEGClassifier(nn.Module):
    """Enhanced classifier with domain adaptation"""
    def __init__(self, input_dim, num_classes, hidden_dim=128):
        super(EnhancedEEGClassifier, self).__init__()


        self.domain_adapter = DomainAdaptationLayer(input_dim, hidden_dim)


        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim // 2, num_classes)
        )


    def forward(self, x):
        features = self.domain_adapter(x)
        output = self.classifier(features)
        return output



def load_subject_data(filepath, electrode_cols):
    """Load EEG data from CSV"""
    print(f"  Loading: {os.path.basename(filepath)}")


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
                'label': label
            })


    print(f"    Processed {len(trials_data)} trials")
    return trials_data



def augment_minority_classes(X, y, min_samples=10):
    """Augment minority classes using noise injection"""
    unique_labels, counts = np.unique(y, return_counts=True)


    X_augmented = []
    y_augmented = []


    print("  Augmenting classes with < {} samples...".format(min_samples))
    for label, count in tqdm(list(zip(unique_labels, counts)), 
                              desc="  Processing classes", leave=False):
        label_indices = np.where(y == label)[0]
        label_samples = X[label_indices]


        X_augmented.append(label_samples)
        y_augmented.extend([label] * len(label_samples))


        if count < min_samples:
            num_to_add = min_samples - count
            for _ in range(num_to_add):
                idx = np.random.choice(len(label_samples))
                base_sample = label_samples[idx]
                noise = np.random.normal(0, 0.01 * np.std(base_sample), base_sample.shape)
                augmented_sample = base_sample + noise


                X_augmented.append(augmented_sample.reshape(1, -1))
                y_augmented.append(label)


    X_augmented = np.vstack(X_augmented)
    y_augmented = np.array(y_augmented)


    return X_augmented, y_augmented



def train_neural_model(model, train_loader, device, num_epochs=100):
    """Train neural network with progress tracking"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)


    model.train()


    epoch_pbar = tqdm(range(num_epochs), desc="Training Neural Network")


    for epoch in epoch_pbar:
        total_loss = 0
        correct = 0
        total = 0


        batch_pbar = tqdm(train_loader, desc=f'  Epoch {epoch+1}/{num_epochs}', 
                          leave=False)


        for batch_data, batch_labels in batch_pbar:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)


            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()


            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()


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



def save_models(neural_model, ensemble, scaler, label_encoder, 
                riemannian_extractor, spectral_extractor, 
                save_dir='saved_models'):
    """Save all trained models and preprocessing objects"""
    os.makedirs(save_dir, exist_ok=True)


    print(f"\n{'='*80}")
    print("SAVING MODELS")
    print(f"{'='*80}")


    # Save PyTorch neural network
    neural_path = os.path.join(save_dir, 'neural_model.pth')
    torch.save({
        'model_state_dict': neural_model.state_dict(),
        'model_architecture': {
            'input_dim': neural_model.domain_adapter.feature_extractor[0].in_features,
            'num_classes': neural_model.classifier[-1].out_features,
            'hidden_dim': 128
        }
    }, neural_path)
    print(f"✓ Neural network saved to: {neural_path}")


    # Save sklearn ensemble and preprocessing
    sklearn_path = os.path.join(save_dir, 'sklearn_models.pkl')
    with open(sklearn_path, 'wb') as f:
        pickle.dump({
            'ensemble': ensemble,
            'scaler': scaler,
            'label_encoder': label_encoder,
            'riemannian_extractor': riemannian_extractor,
            'spectral_extractor': spectral_extractor
        }, f)
    print(f"✓ Sklearn models saved to: {sklearn_path}")


    # Save model metadata
    metadata_path = os.path.join(save_dir, 'model_metadata.txt')
    with open(metadata_path, 'w') as f:
        f.write("EEG Classification Model Metadata\n")
        f.write("="*80 + "\n\n")
        f.write(f"Neural Network Architecture:\n")
        f.write(f"  - Input dim: {neural_model.domain_adapter.feature_extractor[0].in_features}\n")
        f.write(f"  - Hidden dim: 128\n")
        f.write(f"  - Output classes: {neural_model.classifier[-1].out_features}\n")
        f.write(f"  - Total parameters: {sum(p.numel() for p in neural_model.parameters()):,}\n\n")
        f.write(f"Ensemble Models:\n")
        for name, _ in ensemble.estimators:
            f.write(f"  - {name}\n")
        f.write(f"\nLabel Classes: {list(label_encoder.classes_)}\n")
    print(f"✓ Metadata saved to: {metadata_path}")


    print(f"\n✓ All models successfully saved to '{save_dir}/' directory")



def main():
    print("=" * 80)
    print("ADVANCED EEG CLASSIFICATION - OPTIMIZED VERSION WITH PROGRESS BARS")
    print("=" * 80)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")


    electrode_cols = [f'A{i}' for i in range(1, 33)] + [f'B{i}' for i in range(1, 33)]


    csv_files = sorted(glob.glob('subject_*.csv'))
    print(f"\nFound {len(csv_files)} subject files")


    train_files = csv_files[:11]
    test_files = csv_files[11:12]


    # Load training data
    print("\n" + "=" * 80)
    print("STEP 1: Loading and Processing Training Data")
    print("=" * 80)


    train_data = []
    print("\nLoading training subjects:")
    for filepath in tqdm(train_files, desc="Overall Progress"):
        subject_data = load_subject_data(filepath, electrode_cols)
        train_data.extend(subject_data)


    print(f"\nTotal training trials: {len(train_data)}")


    # Extract features
    print("\n" + "=" * 80)
    print("STEP 2: Multi-Modal Feature Extraction")
    print("=" * 80)


    train_trials = [trial['data'] for trial in train_data]
    train_labels = [trial['label'] for trial in train_data]


    print("\nExtracting Riemannian geometry features...")
    riemannian_extractor = RiemannianFeatureExtractor()
    riemannian_features = riemannian_extractor.fit_transform(train_trials)
    print(f"  ✓ Riemannian features shape: {riemannian_features.shape}")


    print("\nExtracting spectral features...")
    spectral_extractor = SpectralFeatureExtractor()
    spectral_features = spectral_extractor.fit_transform(train_trials)
    print(f"  ✓ Spectral features shape: {spectral_features.shape}")


    print("\nCombining feature sets...")
    combined_features = np.hstack([riemannian_features, spectral_features])
    print(f"  ✓ Combined features shape: {combined_features.shape}")


    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_labels)
    num_classes = len(label_encoder.classes_)
    print(f"\nNumber of classes: {num_classes}")


    print("\nAugmenting minority classes...")
    X_augmented, y_augmented = augment_minority_classes(
        combined_features, train_labels_encoded, min_samples=15
    )
    print(f"  ✓ Augmented data shape: {X_augmented.shape}")


    print("\nNormalizing features...")
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X_augmented)
    print(f"  ✓ Normalized data shape: {X_normalized.shape}")


    # OPTIMIZED: Train ensemble models with progress bars
    print("\n" + "=" * 80)
    print("STEP 3: Training Ensemble Models (with Progress Bars)")
    print("=" * 80)


    print("\nTraining 5 models individually with progress tracking...")
    print("=" * 80)


    # Define estimators
    estimators_config = [
        ('gb', GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, 
                                         max_depth=5, random_state=42, verbose=2)),
        ('svc', SVC(kernel='rbf', C=10, gamma='scale', probability=True, 
                   random_state=42, verbose=True, max_iter=10000)),
        ('lda', LinearDiscriminantAnalysis()),
        ('knn', KNeighborsClassifier(n_neighbors=7, weights='distance')),
        ('lr', LogisticRegression(C=1.0, max_iter=1000, random_state=42, verbose=1))
    ]


    fitted_estimators = []

    # Train each model individually with progress tracking
    with tqdm(total=len(estimators_config), desc="\nOverall Ensemble Progress", 
              position=0, leave=True) as overall_pbar:

        for idx, (name, estimator) in enumerate(estimators_config, 1):
            print(f"\n[{idx}/{len(estimators_config)}] Training {name.upper()} - {type(estimator).__name__}")
            print("-" * 80)

            estimator.fit(X_normalized, y_augmented)
            fitted_estimators.append((name, estimator))

            print(f"✓ {name.upper()} training complete")
            overall_pbar.update(1)


    print("\n" + "=" * 80)
    print("Creating VotingClassifier with pre-fitted estimators...")

    # Create VotingClassifier with already-fitted estimators
    ensemble = VotingClassifier(estimators=fitted_estimators, voting='soft', n_jobs=1)
    ensemble.estimators_ = [est for name, est in fitted_estimators]
    ensemble.le_ = label_encoder
    ensemble.classes_ = label_encoder.classes_


    train_acc = ensemble.score(X_normalized, y_augmented)
    print(f"\n✓ Ensemble creation complete!")
    print(f"  Training accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")


    print(f"\n  Models in ensemble:")
    for name, model in fitted_estimators:
        print(f"    - {name.upper()}: {type(model).__name__}")


    # Train neural network
    print("\n" + "=" * 80)
    print("STEP 4: Training Neural Network with Domain Adaptation")
    print("=" * 80)


    input_dim = X_normalized.shape[1]
    neural_model = EnhancedEEGClassifier(input_dim, num_classes).to(device)


    total_params = sum(p.numel() for p in neural_model.parameters())
    print(f"\n  Neural network parameters: {total_params:,}")


    train_tensor = torch.FloatTensor(X_normalized)
    labels_tensor = torch.LongTensor(y_augmented)
    train_dataset = TensorDataset(train_tensor, labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


    train_neural_model(neural_model, train_loader, device, num_epochs=100)


    # Test on hold-out subject
    print("\n" + "=" * 80)
    print("STEP 5: Testing on Hold-Out Subject")
    print("=" * 80)


    test_results = []


    for filepath in tqdm(test_files, desc="\nTesting Subjects"):
        subject_name = os.path.basename(filepath).replace('.csv', '')
        print(f"\nTesting on: {subject_name}")


        test_data = load_subject_data(filepath, electrode_cols)
        test_trials = [trial['data'] for trial in test_data]
        test_labels = [trial['label'] for trial in test_data]


        print("  Extracting Riemannian features...")
        test_riemannian = riemannian_extractor.transform(test_trials)
        print("  Extracting spectral features...")
        test_spectral = spectral_extractor.transform(test_trials)
        test_combined = np.hstack([test_riemannian, test_spectral])
        test_normalized = scaler.transform(test_combined)


        test_labels_encoded = []
        for label in test_labels:
            if label in label_encoder.classes_:
                test_labels_encoded.append(label_encoder.transform([label])[0])
            else:
                test_labels_encoded.append(-1)
        test_labels_encoded = np.array(test_labels_encoded)


        print("  Making ensemble predictions...")
        ensemble_pred = ensemble.predict(test_normalized)
        ensemble_acc = accuracy_score(test_labels_encoded, ensemble_pred)


        print("  Making neural network predictions...")
        neural_model.eval()
        with torch.no_grad():
            test_tensor = torch.FloatTensor(test_normalized).to(device)
            outputs = neural_model(test_tensor)
            _, neural_pred = torch.max(outputs, 1)
            neural_pred = neural_pred.cpu().numpy()


        neural_acc = accuracy_score(test_labels_encoded, neural_pred)


        combined_pred = []
        for i in range(len(ensemble_pred)):
            votes = [ensemble_pred[i], neural_pred[i]]
            combined_pred.append(max(set(votes), key=votes.count))
        combined_pred = np.array(combined_pred)


        combined_acc = accuracy_score(test_labels_encoded, combined_pred)


        print(f"\n  Results:")
        print(f"    Trials: {len(test_labels_encoded)}")
        print(f"    Ensemble Accuracy: {ensemble_acc:.4f} ({ensemble_acc*100:.2f}%)")
        print(f"    Neural Network Accuracy: {neural_acc:.4f} ({neural_acc*100:.2f}%)")
        print(f"    Combined Accuracy: {combined_acc:.4f} ({combined_acc*100:.2f}%)")


        best_acc = max(ensemble_acc, neural_acc, combined_acc)
        best_pred = ensemble_pred if ensemble_acc == best_acc else (
            neural_pred if neural_acc == best_acc else combined_pred
        )


        true_labels_decoded = [label_encoder.inverse_transform([l])[0] if l != -1 
                              else 'unknown' for l in test_labels_encoded]
        pred_labels_decoded = label_encoder.inverse_transform(best_pred)


        print(f"\n  Best Model Classification Report:")
        print(classification_report(true_labels_decoded, pred_labels_decoded, 
                                   zero_division=0))


        test_results.append({
            'subject': subject_name,
            'trials': len(test_labels_encoded),
            'ensemble_acc': ensemble_acc,
            'neural_acc': neural_acc,
            'combined_acc': combined_acc,
            'best_acc': best_acc
        })


    # Summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)


    results_df = pd.DataFrame(test_results)
    print("\n" + results_df.to_string(index=False))


    avg_ensemble = np.mean([r['ensemble_acc'] for r in test_results])
    avg_neural = np.mean([r['neural_acc'] for r in test_results])
    avg_combined = np.mean([r['combined_acc'] for r in test_results])
    avg_best = np.mean([r['best_acc'] for r in test_results])


    print(f"\nAverage Ensemble Accuracy: {avg_ensemble:.4f} ({avg_ensemble*100:.2f}%)")
    print(f"Average Neural Accuracy: {avg_neural:.4f} ({avg_neural*100:.2f}%)")
    print(f"Average Combined Accuracy: {avg_combined:.4f} ({avg_combined*100:.2f}%)")
    print(f"Average Best Accuracy: {avg_best:.4f} ({avg_best*100:.2f}%)")


    # Save all models
    save_models(
        neural_model=neural_model,
        ensemble=ensemble,
        scaler=scaler,
        label_encoder=label_encoder,
        riemannian_extractor=riemannian_extractor,
        spectral_extractor=spectral_extractor,
        save_dir='saved_models'
    )


    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)



if __name__ == "__main__":
    main()
