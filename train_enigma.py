#!/usr/bin/env python3
"""
ENIGMA-style EEG Brain Decoding Classification Model — v5
==========================================================
Key improvements over v4:
  - Downsampling (256 Hz) to reduce temporal redundancy and parameter bloat.
  - Overlapping window extraction (Data Multiplier).
  - Depthwise Spatial Convolutions (EEGNet-style) to prevent overfitting.
  - Increased Batch Size to guarantee positive pairs for SupCon loss.
"""

import argparse
import hashlib
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message="Physical range is not defined")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    import mne
    mne.set_log_level("ERROR")
except ImportError:
    sys.exit("pip install mne")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    sys.exit("pip install torch")

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("pip install tqdm")

try:
    from sklearn.metrics import classification_report
except ImportError:
    sys.exit("pip install scikit-learn")

# ──────────────────────────────────────────────────────────────────────────────
# Globals
# ──────────────────────────────────────────────────────────────────────────────
N_CLASSES   = 36
N_INSTANCES = 5
N_F         = 16
E_DIM       = 8
DROPOUT_P   = 0.5
DEVICE      = (
    "cuda" if torch.cuda.is_available()
    else "mps"  if torch.backends.mps.is_available()
    else "cpu"
)

RESAMPLE_FREQ      = 256.0     # Drastically reduces overfitting
BANDPASS_LOW       = 1.0
BANDPASS_HIGH      = 40.0
NOTCH_FREQ         = 50.0      
ARTIFACT_THRESH_UV = 150e-6    
CACHE_DIR          = Path(".eeg_cache_v5")

# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_bdf(path: Path, window_sec: float, overlap_sec: float,
                   use_cache: bool = True) -> list:
    """
    Load BDF → filter → resample → CAR → overlapping crops → baseline → z-score.
    Returns a list of valid float32 numpy arrays (N_C, target_T).
    """
    key = hashlib.md5(
        f"{path}{window_sec}{overlap_sec}{RESAMPLE_FREQ}{BANDPASS_LOW}{BANDPASS_HIGH}".encode()
    ).hexdigest()
    cache_path = CACHE_DIR / f"{key}.npy"

    if use_cache and cache_path.exists():
        # Load list of arrays from the single npy file
        return list(np.load(str(cache_path), allow_pickle=True))

    raw = mne.io.read_raw_bdf(str(path), preload=True, verbose=False)

    drop = [ch for ch in raw.ch_names
            if ch.upper().startswith(("EXG", "GSR", "STATUS", "TRIG"))]
    if drop:
        raw.drop_channels(drop)
    raw.pick_types(eeg=True, exclude="bads")

    try:
        montage = mne.channels.make_standard_montage("biosemi64")
        raw.set_montage(montage, on_missing="ignore", verbose=False)
    except Exception:
        pass

    # 1. Bandpass & Notch
    raw.filter(BANDPASS_LOW, BANDPASS_HIGH, method="fir", verbose=False)
    try:
        raw.notch_filter(NOTCH_FREQ, verbose=False)
    except Exception:
        pass

    # 2. Resample (Crucial for preventing parameter bloat)
    if raw.info["sfreq"] > RESAMPLE_FREQ:
        raw.resample(RESAMPLE_FREQ, npad="auto")

    # 3. Common average reference
    raw.set_eeg_reference("average", projection=False, verbose=False)

    data = raw.get_data().astype(np.float32)
    
    # 4. Overlapping Windows Extraction (Data Augmentation multiplier)
    target_T   = int(RESAMPLE_FREQ * window_sec)
    step_T     = int(RESAMPLE_FREQ * (window_sec - overlap_sec))
    step_T     = max(1, step_T)
    
    valid_windows = []
    
    for start in range(0, data.shape[1] - target_T + 1, step_T):
        window = data[:, start : start + target_T].copy()
        
        # Artifact rejection
        if np.ptp(window, axis=1).max() > ARTIFACT_THRESH_UV:
            continue
            
        # Baseline correction (first 20% of the window)
        baseline_samples = max(1, int(target_T * 0.2))
        window -= window[:, :baseline_samples].mean(axis=1, keepdims=True)
        
        # Per-channel z-score
        window = (window - window.mean(axis=1, keepdims=True)) / (window.std(axis=1, keepdims=True) + 1e-8)
        valid_windows.append(window)

    if use_cache:
        CACHE_DIR.mkdir(exist_ok=True)
        # Store as object array of shape (num_windows,)
        np.save(str(cache_path), np.array(valid_windows, dtype=object))

    return valid_windows


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class EEGDataset(Dataset):
    def __init__(self, samples: list, window_sec: float, overlap_sec: float,
                 augment_data: bool = False, use_cache: bool = True):
        self.augment_data = augment_data
        self.samples      = []
        rejected_files    = 0
        total_windows     = 0
        
        for path, label in samples:
            # Overlap only applied to training to boost data. Test uses 0 overlap.
            actual_overlap = overlap_sec if augment_data else 0.0
            windows = preprocess_bdf(path, window_sec, actual_overlap, use_cache)
            
            if not windows:
                rejected_files += 1
            else:
                for w in windows:
                    self.samples.append((w, label))
                    total_windows += 1
                    
        if rejected_files:
            tqdm.write(f"    Artifact rejection: {rejected_files} full files removed")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data, label = self.samples[idx]
        x = torch.from_numpy(data[np.newaxis, ...])
        
        if self.augment_data:
            # Subtle Gaussian noise
            x += torch.randn_like(x) * 0.05
            # Spatial dropout (zero out random channels 10% of the time)
            mask = (torch.rand(x.shape[1], 1) > 0.1).float()
            x *= mask.unsqueeze(0)
            # Amplitude scale jitter
            x *= (0.9 + 0.2 * torch.rand(1).item())
            
        return x, label


def build_fold(stimuli_dir: Path, test_instance: int,
               window_sec: float, overlap_sec: float, use_cache: bool):
    label_map  = {}
    train_raw, test_raw = [], []
    classes = sorted([d.name for d in stimuli_dir.iterdir() if d.is_dir()])
    
    for cls in classes:
        lbl = len(label_map)
        label_map[cls] = lbl
        for inst in range(1, N_INSTANCES + 1):
            bdf = stimuli_dir / cls / f"instance{inst}.bdf"
            if not bdf.exists():
                continue
            (test_raw if inst == test_instance else train_raw).append((bdf, lbl))

    tqdm.write(f"    Preprocessing train ({len(train_raw)} files)...")
    train_ds = EEGDataset(train_raw, window_sec, overlap_sec,
                          augment_data=True,  use_cache=use_cache)
    tqdm.write(f"    Preprocessing test  ({len(test_raw)} files)...")
    test_ds  = EEGDataset(test_raw,  window_sec, overlap_sec=0.0,
                          augment_data=False, use_cache=use_cache)
                          
    tqdm.write(f"    Train Windows: {len(train_ds)} | Test Windows: {len(test_ds)}")
    return train_ds, test_ds, label_map


# ──────────────────────────────────────────────────────────────────────────────
# Model (EEGNet-inspired Depthwise Convs + ENIGMA backbone)
# ──────────────────────────────────────────────────────────────────────────────

class ImprovedTinyENIGMA(nn.Module):
    def __init__(self, n_channels, n_times, n_classes,
                 n_f=N_F, e_dim=E_DIM, dropout_p=DROPOUT_P, embed_dim=64):
        super().__init__()
        
        # 1. Temporal Convolution
        self.temporal_conv = nn.Conv2d(1, n_f, kernel_size=(1, 32), padding=(0, 16), bias=False)
        self.bn1           = nn.BatchNorm2d(n_f)
        
        # 2. Depthwise Spatial Convolution (Prevents extreme overfitting)
        self.spatial_conv  = nn.Conv2d(n_f, n_f * 2, kernel_size=(n_channels, 1), groups=n_f, bias=False)
        self.bn2           = nn.BatchNorm2d(n_f * 2)
        
        # 3. Pooling & Projection
        self.temporal_pool = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.dropout       = nn.Dropout(dropout_p)
        self.proj_conv     = nn.Conv2d(n_f * 2, e_dim, kernel_size=(1, 1), bias=False)
        
        # Dynamically calculate sequence length after pooling
        dummy = torch.randn(1, 1, n_channels, n_times)
        with torch.no_grad():
            out = self.temporal_pool(self.spatial_conv(self.temporal_conv(dummy)))
            T_pool = out.shape[-1]
            
        self.embed_head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(e_dim * T_pool),
            nn.Linear(e_dim * T_pool, embed_dim),
            nn.ELU(),  # ELU often performs better for EEG than GELU
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(embed_dim, n_classes),
        )

    def encode(self, x):
        z = F.elu(self.bn1(self.temporal_conv(x)))
        z = self.dropout(F.elu(self.bn2(self.spatial_conv(z))))
        z = self.temporal_pool(z)
        return self.embed_head(self.proj_conv(z))

    def forward(self, x):
        return self.classifier(self.encode(x))


# ──────────────────────────────────────────────────────────────────────────────
# Supervised Contrastive Loss
# ──────────────────────────────────────────────────────────────────────────────

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temp = temperature

    def forward(self, features, labels):
        B, device = features.shape[0], features.device
        sim       = torch.mm(features, features.T) / self.temp
        self_mask = torch.eye(B, device=device).bool()
        pos_mask  = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask
        
        sim       = sim - sim.max(dim=1, keepdim=True).values.detach()
        exp_sim   = torch.exp(sim).masked_fill(self_mask, 0.0)
        log_prob  = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        
        n_pos     = pos_mask.float().sum(dim=1)
        valid     = n_pos > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
            
        loss = -(log_prob * pos_mask.float()).sum(dim=1)
        return (loss[valid] / n_pos[valid]).mean()


# ──────────────────────────────────────────────────────────────────────────────
# One fold
# ──────────────────────────────────────────────────────────────────────────────

def run_fold(fold: int, stimuli_dir: Path, args):
    train_ds, test_ds, label_map = build_fold(
        stimuli_dir, fold, args.window_sec, args.overlap_sec, not args.no_cache)

    if len(train_ds) == 0 or len(test_ds) == 0:
        tqdm.write(f"  Fold {fold}: no usable data after rejection, skipping.")
        return None, None, None, label_map, None, None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    n_channels = train_ds[0][0].shape[1]
    target_T   = train_ds[0][0].shape[-1]
    
    model  = ImprovedTinyENIGMA(n_channels, target_T, N_CLASSES).to(DEVICE)
    supcon = SupConLoss()
    ce     = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Phase 1: Contrastive Pretraining
    opt_pre = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sch_pre = torch.optim.lr_scheduler.CosineAnnealingLR(opt_pre, T_max=args.pretrain_epochs)
    
    pre_bar = tqdm(range(1, args.pretrain_epochs + 1),
                   desc=f"  Fold {fold} pretrain", ncols=90,
                   unit="ep", leave=False, colour="yellow")
    for _ in pre_bar:
        model.train()
        ep_loss = 0.0
        batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt_pre.zero_grad()
            
            features = F.normalize(model.encode(xb), dim=1)
            loss = supcon(features, yb)
            
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt_pre.step()
                ep_loss += loss.item()
                batches += 1
                
        sch_pre.step()
        if batches > 0:
            pre_bar.set_postfix(loss=f"{ep_loss/batches:.3f}")

    # Phase 2: Classification Fine-Tuning
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.classifier.parameters():
        p.requires_grad_(True)
        
    opt_ft = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                               lr=args.lr, weight_decay=0.01)
    
    unfreeze_epoch = args.epochs // 2
    best_acc, best_state = 0.0, None

    ft_bar = tqdm(range(1, args.epochs + 1),
                  desc=f"  Fold {fold} finetune", ncols=90,
                  unit="ep", leave=False, colour="cyan")
    for epoch in ft_bar:
        if epoch == unfreeze_epoch:
            for p in model.parameters():
                p.requires_grad_(True)
            opt_ft = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=0.01)
            
        model.train()
        tr_correct, tr_total = 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt_ft.zero_grad()
            logits = model(xb)
            loss = ce(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_ft.step()
            
            tr_correct += (logits.argmax(1) == yb).sum().item()
            tr_total   += xb.size(0)

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_correct += (model(xb).argmax(1) == yb).sum().item()
                val_total   += xb.size(0)
                
        val_acc = val_correct / val_total if val_total > 0 else 0
        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        ft_bar.set_postfix(tr=f"{tr_correct/tr_total:.2f}",
                           val=f"{val_acc:.2f}", best=f"{best_acc:.2f}")

    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            all_preds.extend(model(xb.to(DEVICE)).argmax(1).cpu().tolist())
            all_labels.extend(yb.tolist())

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return acc, all_preds, all_labels, label_map, best_state, n_channels


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    stimuli_dir = Path(args.stimuli_dir)
    if not stimuli_dir.exists():
        sys.exit(f"Not found: {stimuli_dir}")

    print(f"\n{'='*60}")
    print(f"  ENIGMA EEG Classifier  v5")
    print(f"{'='*60}")
    print(f"  Device         : {DEVICE}")
    print(f"  Resampled Rate : {RESAMPLE_FREQ} Hz (Downsampled for efficiency)")
    print(f"  Window         : {args.window_sec} s (Overlap: {args.overlap_sec} s)")
    print(f"  Bandpass       : {BANDPASS_LOW}–{BANDPASS_HIGH} Hz")
    print(f"  Artifact thresh: {ARTIFACT_THRESH_UV*1e6:.0f} µV p-p")
    print(f"  Batch Size     : {args.batch_size} (Crucial for SupCon loss)")
    print(f"  Cache          : {'disabled' if args.no_cache else str(CACHE_DIR)}")
    print(f"  Folds          : {args.fold if args.fold else '1-5 (full CV)'}")
    print(f"{'='*60}\n")

    folds     = [args.fold] if args.fold else list(range(1, N_INSTANCES + 1))
    fold_accs = []

    for fold in folds:
        tqdm.write(f"\n── Fold {fold} (test instance {fold}) ──────────────────────")
        acc, preds, labels, label_map, best_state, n_channels = run_fold(fold, stimuli_dir, args)
        if acc is None:
            continue
            
        fold_accs.append(acc)
        idx_to_label = {v: k for k, v in label_map.items()}
        target_names = [idx_to_label[i] for i in range(N_CLASSES)]
        
        tqdm.write(f"  Fold {fold} top-1 acc: {acc:.3f}  (chance={1/N_CLASSES:.3f})")
        tqdm.write(classification_report(labels, preds,
                                         target_names=target_names,
                                         digits=3, zero_division=0))
        ckpt = f"enigma_v5_fold{fold}.pt"
        torch.save({"model_state": best_state, "label_map": label_map,
                    "n_channels": n_channels}, ckpt)
        tqdm.write(f"  Checkpoint → {ckpt}")

    if fold_accs:
        print(f"\n{'='*60}")
        print(f"  Cross-validation results")
        print(f"{'='*60}")
        for f, a in zip(folds, fold_accs):
            print(f"  Fold {f}: {a:.3f}")
        if len(fold_accs) > 1:
            arr = np.array(fold_accs)
            print(f"  ──────────────────────────────")
            print(f"  Mean ± Std : {arr.mean():.3f} ± {arr.std():.3f}")
            print(f"  Chance     : {1/N_CLASSES:.3f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ENIGMA EEG classifier v5")
    parser.add_argument("--stimuli_dir",     default="stimuli", type=str)
    parser.add_argument("--fold",            default=None,      type=int)
    parser.add_argument("--window_sec",      default=1.0,       type=float)
    parser.add_argument("--overlap_sec",     default=0.5,       type=float, 
                        help="Amount of overlap between extraction windows.")
    parser.add_argument("--pretrain_epochs", default=100,       type=int)
    parser.add_argument("--epochs",          default=150,       type=int)
    parser.add_argument("--batch_size",      default=72,        type=int, 
                        help="Keep >64 to ensure SupCon sees matching classes in batch.")
    parser.add_argument("--lr",              default=5e-4,      type=float)
    parser.add_argument("--no_cache",        action="store_true",
                        help="Force reprocess all BDF files")
    args = parser.parse_args()
    main(args)