"""
Handles everything that happens during training:
- One epoch of training (forward pass, loss, backward pass, optimizer step)
- One epoch of validation
- Early stopping
- Learning rate scheduling
- Saving the best model checkpoint
- Tracking metrics (loss, accuracy) over all epochs
"""
 
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader as DL
 
from src.data.loader   import process_files, MEGDataset, _worker_init
from src.config.config import *
 
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

def count_params(model: nn.Module) -> int:
    """Return the number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def mixup_batch(
        X: torch.Tensor,
        y: torch.Tensor,
        alpha: float,
        num_classes: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply MixUp to a single batch.
 
    For each sample i, pick a random partner j and blend:
        X_mixed = λ * X[i] + (1-λ) * X[j]
        y_mixed = λ * onehot(y[i]) + (1-λ) * onehot(y[j])
 
    λ ~ Beta(alpha, alpha).  When alpha = 0.0 this is skipped entirely.
    When alpha is small (e.g. 0.2) most λ values are close to 0 or 1 so
    most samples are barely changed — just enough to smooth boundaries.
 
    Returns mixed X (float) and soft one-hot y (float) for use with
    cross_entropy against soft targets.
    """
    lam = float(np.random.beta(alpha, alpha))
    B   = X.size(0)
    idx = torch.randperm(B, device = X.device)
 
    X_mix = lam * X + (1.0 - lam) * X[idx]
 
    # Soft labels
    y_a = F.one_hot(y,        num_classes).float()
    y_b = F.one_hot(y[idx],   num_classes).float()
    y_mix = lam * y_a + (1.0 - lam) * y_b
 
    return X_mix, y_mix
 
def mixup_loss(
        criterion: nn.CrossEntropyLoss,
        logits:    torch.Tensor,
        y_soft:    torch.Tensor
    ) -> torch.Tensor:
    """
    CrossEntropyLoss with soft (mixed) targets.
    nn.CrossEntropyLoss accepts float target tensors of shape (B, C) directly
    as of PyTorch 1.10+, applying the same label-smoothing if set.
    """
    return criterion(logits, y_soft)

# One training epoch
def train_epoch(
        model:       nn.Module,
        loader:      DataLoader,
        optimizer:   torch.optim.Optimizer,
        criterion:   nn.CrossEntropyLoss,
        device:      str,
        grad_clip:   float = GRAD_CLIP,
        mixup_alpha: float = MIXUP_ALPHA_INTRA
    ) -> tuple[float, float]:
    """
    One full pass over the training set.
    If mixup_alpha > 0, MixUp is applied to every batch.
    Accuracy is computed on the original (un-mixed) labels for interpretability.
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0
 
    for X, y in loader:
        X, y = X.to(device), y.to(device)
 
        if mixup_alpha > 0.0:
            X_in, y_soft = mixup_batch(X, y, mixup_alpha, NUM_CLASSES)
            logits = model(X_in)
            loss   = mixup_loss(criterion, logits, y_soft)
        else:
            logits = model(X)
            loss   = criterion(logits, y)
 
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
 
        total_loss += loss.item() * len(y)
        # Accuracy always against original hard labels
        correct    += (logits.argmax(1) == y).sum().item()
        total      += len(y)
 
    return total_loss / total, correct / total

# One validation / test epoch
@torch.no_grad()
def eval_epoch(
        model:     nn.Module,
        loader:    DataLoader,
        criterion: nn.Module,
        device:    str
    ) -> tuple[float, float]:
    """Evaluate on a DataLoader. No MixUp during validation/test"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
 
    for X, y in loader:
        X, y   = X.to(device), y.to(device)
        logits = model(X)
        loss   = criterion(logits, y)
 
        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += len(y)
 
    return total_loss / total, correct / total

@torch.no_grad()
def get_predictions(
        model:  nn.Module,
        loader: DataLoader,
        device: str
    ) -> tuple[list[int], list[int]]:
    """Collect all ground-truth labels and model predictions"""
    model.eval()
    y_true, y_pred = [], []
 
    for X, y in loader:
        preds = model(X.to(device)).argmax(1).cpu().tolist()
        y_true.extend(y.tolist())
        y_pred.extend(preds)
 
    return y_true, y_pred

def setup_training(
        model:           nn.Module,
        lr:              float,
        weight_decay:    float,
        epochs:          int,
        label_smoothing: float,
    ):
    if OPTIMIZER == "adam_w":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr = lr, weight_decay = weight_decay)
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr = lr, weight_decay = weight_decay)
 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max = epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing = label_smoothing)
 
    return optimizer, scheduler, criterion

# Full training run with early stopping
def train(
        model:           nn.Module,
        train_loader:    DataLoader,
        val_loader:      DataLoader,
        model_name:      str   = "model",
        experiment:      str   = "intra",
        epochs:          int   = EPOCHS,
        lr:              float = LEARNING_RATE,
        weight_decay:    float = WEIGHT_DECAY,
        patience:        int   = PATIENCE,
        label_smoothing: float = LABEL_SMOOTHING,
        mixup_alpha:     float = MIXUP_ALPHA_INTRA
    ) -> dict[str, list[float]]:
    """Full training loop with early stopping and checkpoint saving"""
    model    = model.to(DEVICE)
    n_params = count_params(model)
    print(f"\n--Device       : {DEVICE}")
    print(f"--Model        : {model_name}  ({n_params:,} parameters)")
    print(f"--Experiment   : {experiment}")
    print(f"--Epochs (max) : {epochs} with Patience: {patience}")
    print(f"--MixUp alpha  : {mixup_alpha}")
 
    optimizer, scheduler, criterion = setup_training(
        model, lr, weight_decay, epochs, label_smoothing)
 
    ckpt_path = CHECKPOINTS_DIR / f"{model_name}_{experiment}_best.pt"
 
    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  []}
 
    best_val_loss     = float("inf")
    epochs_no_improve = 0
    end_early_epoch   = None
 
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, DEVICE,
            mixup_alpha = mixup_alpha)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, DEVICE)
        scheduler.step()
 
        history["train_loss"].append(train_loss)
        history["val_loss"  ].append(val_loss)
        history["train_acc" ].append(train_acc)
        history["val_acc"   ].append(val_acc)
 
        if epoch % 5 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch:3d}/{epochs}"
                f"  | train loss {train_loss:.4f}  acc {train_acc:.3f}"
                f"  | val loss {val_loss:.4f}  acc {val_acc:.3f}"
                f"  | lr {lr_now:.2e}"
            )
 
        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n-- Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                end_early_epoch = epoch
                break
 
    model.load_state_dict(torch.load(ckpt_path, map_location = DEVICE))
    print(f"\nSaved the best model to {ckpt_path}")
    history["last-epoch"] = [float(end_early_epoch or epochs)]
    return history

# Chunked training for Cross experiment (memory management)
def train_chunked(
        model:           nn.Module,
        file_chunks:     list[list[Path]],
        val_loader:      DataLoader,
        model_name:      str   = "model",
        epochs:          int   = EPOCHS,
        lr:              float = LEARNING_RATE,
        weight_decay:    float = WEIGHT_DECAY,
        patience:        int   = PATIENCE,
        label_smoothing: float = LABEL_SMOOTHING,
        mixup_alpha:     float = MIXUP_ALPHA_CROSS
    ) -> dict[str, list[float]]:
    """
    Training loop for the Cross experiment (chunked file loading).
    MixUp is applied inside the per-chunk batch loop.
    Fixed: scheduler.step() is now called once per epoch (was twice before).
    """
    model    = model.to(DEVICE)
    n_params = count_params(model)
    print(f"\n--Device       : {DEVICE}")
    print(f"--Model        : {model_name}  ({n_params:,} parameters)")
    print(f"--Experiment   : cross (chunked with {len(file_chunks)} chunks)")
    print(f"--MixUp alpha  : {mixup_alpha}")
 
    optimizer, scheduler, criterion = setup_training(
        model, lr, weight_decay, epochs, label_smoothing)
 
    ckpt_path = CHECKPOINTS_DIR / f"{model_name}_cross_best.pt"
 
    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  []}
 
    best_val_loss     = float("inf")
    epochs_no_improve = 0
    end_early_epoch   = None
 
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0
 
        for chunk in file_chunks:
            X, y = process_files(chunk, verbose = False)
            ds   = MEGDataset(X, y)
            loader = DL(
                ds, batch_size = BATCH_SIZE, shuffle = True, drop_last = True,
                num_workers = NUM_WORKERS, pin_memory = True,
                worker_init_fn = _worker_init,
                persistent_workers = NUM_WORKERS > 0)
 
            for Xb, yb in loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
 
                if mixup_alpha > 0.0:
                    Xb_in, yb_soft = mixup_batch(Xb, yb, mixup_alpha, NUM_CLASSES)
                    logits = model(Xb_in)
                    loss   = mixup_loss(criterion, logits, yb_soft)
                else:
                    logits = model(Xb)
                    loss   = criterion(logits, yb)
 
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
 
                epoch_loss    += loss.item() * len(yb)
                epoch_correct += (logits.argmax(1) == yb).sum().item()
                epoch_total   += len(yb)
 
            del X, y, ds, loader
 
        train_loss = epoch_loss    / epoch_total
        train_acc  = epoch_correct / epoch_total
 
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, DEVICE)
        scheduler.step()
 
        history["train_loss"].append(train_loss)
        history["val_loss"  ].append(val_loss)
        history["train_acc" ].append(train_acc)
        history["val_acc"   ].append(val_acc)
 
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{epochs}"
                f"  | train loss {train_loss:.4f}  acc {train_acc:.3f}"
                f"  | val loss {val_loss:.4f}  acc {val_acc:.3f}"
            )
 
        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n-- Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                end_early_epoch = epoch
                break
 
    model.load_state_dict(torch.load(ckpt_path, map_location = DEVICE))
    print(f"\nSaved the best model to: {ckpt_path}")
    history["last-epoch"] = [float(end_early_epoch or epochs)]
    return history
