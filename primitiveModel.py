import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
eeg_file = '/Volumes/Extreme SSD/downsampled 100hz/sub1-100hz.csv'
random_seed = 42

print("="*60)
print("EEG IMAGERY CLASSIFICATION MODEL")
print("="*60)

# 1. LOAD DATA
print("\n1. Loading data...")
df = pd.read_csv(eeg_file)
print(f"   Loaded {len(df)} rows")
print(f"   Unique trials: {df['trial_num'].nunique()}")

# 2. PREPARE LABELS
print("\n2. Preparing labels...")
# Create binary labels: animal (0) vs tool (1)
df['label'] = df['trial_type'].apply(
    lambda x: 0 if 'animal' in str(x).lower() else (1 if 'tool' in str(x).lower() else -1)
)

# Remove any rows without clear labels
df = df[df['label'] != -1]

# Check label distribution
trials_with_labels = df.groupby('trial_num')['label'].first()
print(f"   Animal trials: {(trials_with_labels == 0).sum()}")
print(f"   Tool trials: {(trials_with_labels == 1).sum()}")

# 3. EXTRACT FEATURES PER TRIAL
print("\n3. Extracting features per trial...")
# Get EEG channel columns (exclude metadata)
metadata_cols = ['trial_num', 'time', 'time_from_trial_start', 'trial_type', 'stim_file', 'label']
eeg_channels = [col for col in df.columns if col not in metadata_cols]
print(f"   Found {len(eeg_channels)} EEG channels")

# Create feature vectors - one per trial
trial_features = []
trial_labels = []
trial_nums = []

for trial_num in df['trial_num'].unique():
    trial_data = df[df['trial_num'] == trial_num]
    
    # Extract features: mean amplitude per channel
    # (You can make this more sophisticated later)
    features = []
    for channel in eeg_channels:
        # Mean amplitude
        features.append(trial_data[channel].mean())
        # Standard deviation (adds variability info)
        features.append(trial_data[channel].std())
        # Max amplitude
        features.append(trial_data[channel].max())
        # Min amplitude
        features.append(trial_data[channel].min())
    
    trial_features.append(features)
    trial_labels.append(trial_data['label'].iloc[0])
    trial_nums.append(trial_num)

X = np.array(trial_features)
y = np.array(trial_labels)

print(f"   Feature matrix shape: {X.shape}")
print(f"   (trials x features per trial)")

# 4. SPLIT INTO TRAIN/TEST
print("\n4. Splitting data...")
X_train, X_test, y_train, y_test, train_trials, test_trials = train_test_split(
    X, y, trial_nums, test_size=0.2, random_state=random_seed, stratify=y
)
print(f"   Training set: {len(X_train)} trials")
print(f"   Test set: {len(X_test)} trials")

# 5. NORMALIZE FEATURES
print("\n5. Normalizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. TRAIN MODELS
print("\n6. Training models...")

# Model 1: Logistic Regression (simplest)
print("\n   Model 1: Logistic Regression")
lr_model = LogisticRegression(random_state=random_seed, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_acc = accuracy_score(y_test, lr_pred)
print(f"   Test Accuracy: {lr_acc:.3f}")

# Model 2: Random Forest (handles non-linearity better)
print("\n   Model 2: Random Forest")
rf_model = RandomForestClassifier(n_estimators=100, random_state=random_seed)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"   Test Accuracy: {rf_acc:.3f}")

# 7. EVALUATE BEST MODEL
print("\n7. Detailed Evaluation (Random Forest)...")
print("\nClassification Report:")
print(classification_report(y_test, rf_pred, target_names=['Animal', 'Tool']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, rf_pred)
print(cm)
print("(Rows = True labels, Columns = Predicted labels)")

# 8. VISUALIZE RESULTS
print("\n8. Creating visualizations...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: Model comparison
axes[0].bar(['Logistic Regression', 'Random Forest'], [lr_acc, rf_acc])
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Model Comparison')
axes[0].set_ylim([0, 1])
axes[0].axhline(y=0.5, color='r', linestyle='--', label='Chance level')
axes[0].legend()

# Plot 2: Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=['Animal', 'Tool'], yticklabels=['Animal', 'Tool'])
axes[1].set_title('Confusion Matrix (Random Forest)')
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('model_results.png', dpi=300, bbox_inches='tight')
print("   Saved visualization to 'model_results.png'")

# 9. FEATURE IMPORTANCE
print("\n9. Top 10 Most Important Features (Random Forest):")
feature_names = []
for channel in eeg_channels:
    feature_names.extend([f"{channel}_mean", f"{channel}_std", f"{channel}_max", f"{channel}_min"])

feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10))

# 10. SUMMARY
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Best Model: Random Forest")
print(f"Test Accuracy: {rf_acc:.1%}")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"\nInterpretation:")
if rf_acc > 0.6:
    print("✓ Model shows above-chance performance!")
    print("  The EEG signals contain information about imagery type.")
else:
    print("⚠ Model performance is near chance (50%).")
    print("  Consider: more data, better features, or different approach.")
print("="*60)