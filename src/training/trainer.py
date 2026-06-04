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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.loader import process_files, MEGDataset
from torch.utils.data import DataLoader as DL
from src.config.config import *

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Utility: count model parameters
def count_params(model: nn.Module) -> int:
    """Return the number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# One training epoch
def train_epoch(
        model:     nn.Module,
        loader:    DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device:    str,
        grad_clip: float = GRAD_CLIP
    ) -> tuple[float, float]:
    """Run one full pass over the training set"""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        # Forward pass
        logits = model(X)
        loss   = criterion(logits, y)
        # Backwards pass
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping -> prevent exploding
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # Update weights
        optimizer.step() 

        # Metric tracking
        total_loss += loss.item() * len(y)
        preds       = logits.argmax(dim=1)
        correct    += (preds == y).sum().item()
        total      += len(y)

    return total_loss / total, correct / total

# One validation / test epoch
@torch.no_grad() # Memory efficient (no need to update)
def eval_epoch(
        model:     nn.Module,
        loader:    DataLoader,
        criterion: nn.Module,
        device:    str
    ) -> tuple[float, float]:
    """
    Evaluate model on a DataLoader (validation or test set)
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for X, y in loader:
        X, y    = X.to(device), y.to(device)
        # Forward pass
        logits  = model(X)
        loss    = criterion(logits, y)
        
        # Metric tracking
        total_loss += loss.item() * len(y)
        preds       = logits.argmax(dim=1)
        correct    += (preds == y).sum().item()
        total      += len(y)

    return total_loss / total, correct / total

# Get all predictions and true labels
@torch.no_grad() # Memory efficient (no need to update)
def get_predictions(
        model:  nn.Module,
        loader: DataLoader,
        device: str
    ) -> tuple[list[int], list[int]]:
    """
    Run model over a DataLoader and collect all predictions and ground-truth labels
    """
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []

    for X, y in loader:
        X = X.to(device)
        preds = model(X).argmax(dim=1).cpu().tolist()
        y_true.extend(y.tolist())
        y_pred.extend(preds)

    return y_true, y_pred

def setup_training(model, lr, weight_decay, epochs, label_smoothing
        ) -> tuple[torch.optim.AdamW | torch.optim.Adam,
                torch.optim.lr_scheduler.CosineAnnealingLR,
                nn.CrossEntropyLoss]:
    """Creates the optimizer, LR-scheduler and Loss function"""
    if OPTIMIZER == "adam_w":
        # AdamW: like Adam but with better weight decay handling
        optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)

    # Cosine annealing: smoothly reduce LR from lr -> 0 over 'epochs' steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs)
    # Label smoothing: not binary but smoothed over [0, 1]
    # Prevents the model from becoming overconfident
    criterion = nn.CrossEntropyLoss(label_smoothing = label_smoothing)

    return optimizer, scheduler, criterion

# Full training run with early stopping
def train(
        model:        nn.Module,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        model_name:   str   = "model",
        experiment:   str   = "intra",
        epochs:       int   = EPOCHS,
        lr:           float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        patience:     int   = PATIENCE,
        label_smoothing: float = 0.1
    ) -> dict[str, list[float]]:
    """
    NOTE: Important
    Full training loop with early stopping and checkpoint saving

    Parameters
    model        : PyTorch model to train
    train_loader : DataLoader for training data
    val_loader   : DataLoader for validation data
    model_name   : string identifier (e.g. "gru")
    experiment   : "intra" or "cross"
    epochs       : maximum number of training epochs
    lr           : learning rate
    weight_decay : L2 regularisation strength
    patience     : early stopping patience (epochs without improvement)
    device_str   : "cuda" or "cpu"
    label_smoothing : softens the target labels slightly (reduces overconfidence)

    Returns
    history : dict with lists of train_loss, val_loss, train_acc, val_acc per epoch
    """
    # Setup
    model = model.to(DEVICE)
    parameters_amount = count_params(model)
    print(f"\n--Device       : {DEVICE}")
    print(f"--Model        : {model_name}  ({parameters_amount:,} parameters)")
    print(f"--Experiment   : {experiment}")
    print(f"--Epochs (max) : {epochs} with Patience: {patience}")

    optimizer, scheduler, criterion = setup_training(
        model, lr, weight_decay, epochs, label_smoothing)

    ckpt_path = CHECKPOINTS_DIR / f"{model_name}_{experiment}_best.pt"

    # Track training history
    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  []}
    # Main epoch loop
    best_val_loss, epochs_no_improve = float("inf"), 0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader,
            optimizer, criterion, DEVICE)
        val_loss,   val_acc   = eval_epoch(model, val_loader,
            criterion, DEVICE)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"  ].append(val_loss)
        history["train_acc" ].append(train_acc)
        history["val_acc"   ].append(val_acc)

        # Print progress every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch:3d}/{epochs}"
                f"  - train loss {train_loss:.4f}  acc {train_acc:.3f}"
                f"  - val loss {val_loss:.4f}  acc {val_acc:.3f}"
                f"  - lr {lr_now:.2e}"
            )

        # Early stopping and model saving
        end_early_epoch: int | None = None
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n--Warning-- \
                    \nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
                end_early_epoch = epoch
                break

    # Load best weights before returning
    model.load_state_dict(torch.load(ckpt_path, map_location = DEVICE))
    print(f"\nSaved the best model to {ckpt_path}")
    history["last-epoch"] = [float(end_early_epoch or epochs)]
    return history

# Chunked training for Cross experiment (memory management)
def train_chunked(
        model:       nn.Module,
        file_chunks: list[list[Path]],
        val_loader:  DataLoader,
        model_name:  str    = "model",
        epochs:      int    = EPOCHS,
        lr:          float  = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        patience:    int    = PATIENCE,
        label_smoothing: float = 0.1
    ) -> dict[str, list[float]]:
    """
    Training loop that processes files in small chunks (hint 3)
    """
    model  = model.to(DEVICE)
    print(f"\n  Device       : {DEVICE}")
    parameters_amount = count_params(model)
    print(f"  Model        : {model_name}  ({parameters_amount:,} parameters)")
    print(f"  Experiment   : cross (chunked, {len(file_chunks)} chunks)")

    optimizer, scheduler, criterion = setup_training(
        model, lr, weight_decay, epochs, label_smoothing)

    ckpt_path = CHECKPOINTS_DIR / f"{model_name}_cross_best.pt"

    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  []}
        
    best_val_loss, epochs_no_improve = float("inf"), 0

    for epoch in range(1, epochs + 1):
        # ── Train: iterate over file chunks ───────────────────────────────
        model.train()
        epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0

        for chunk in file_chunks:
            # Load just this chunk into memory
            X, y = process_files(chunk, verbose = False)
            ds   = MEGDataset(X, y)
            loader = DL(ds, batch_size = BATCH_SIZE, shuffle = True, drop_last = True)

            for Xb, yb in loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                logits  = model(Xb)
                loss    = criterion(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

                epoch_loss    += loss.item() * len(yb)
                epoch_correct += (logits.argmax(1) == yb).sum().item()
                epoch_total   += len(yb)

            # Free chunk from RAM immediately
            del X, y, ds, loader

        train_loss = epoch_loss    / epoch_total
        train_acc  = epoch_correct / epoch_total

        # ── Validate ──────────────────────────────────────────────────────
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, DEVICE)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"  ].append(val_loss)
        history["train_acc" ].append(train_acc)
        history["val_acc"   ].append(val_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{epochs}"
                f"  - train loss {train_loss:.4f}  acc {train_acc:.3f}"
                f"  - val loss {val_loss:.4f}  acc {val_acc:.3f}"
            )

        # Early stopping and model saving
        end_early_epoch: int | None = None
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n--Warning-- \
                    \nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
                end_early_epoch = epoch
                break

    model.load_state_dict(torch.load(ckpt_path, map_location = DEVICE))
    print(f"\nSaved the best model to: {ckpt_path}")
    history["last-epoch"] = [float(end_early_epoch or epochs)]
    return history
