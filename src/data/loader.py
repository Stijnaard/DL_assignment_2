"""
Reads the .h5 files, downsamples (hint 2) and normalises (hint 1)
Cuts the signal into short windows (each window = one training sample)

Normalisatie-strategie:
  Statistieken (mediaan + IQR) worden berekend over ALLE trainingsbestanden
  samen, en daarna bevroren toegepast op train/val/test. Zo blijven de
  amplitude-verhoudingen tussen taken (rest vs. motor vs. math vs. memory)
  bewaard — die zijn namelijk discriminatief voor het model.

  Vroeger werd elk bestand afzonderlijk genormaliseerd, waardoor de absolute
  amplitude per taak werd weggegooid. Dat was het probleem.
"""

from pathlib import Path
import h5py
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from scipy.signal import decimate as scipy_decimate

from src.config.config import *

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _worker_init(worker_id: int):
    """Give each DataLoader worker its own reproducible seed"""
    np.random.seed(SEED + worker_id)


# ---------------------------------------------------------------------------
# 1. Labels
# ---------------------------------------------------------------------------

def get_label(filepath: Path) -> int:
    """
    Parse the class index from the filename prefix.
    'rest_XXXXX_1.h5'              -> 0
    'task_motor_XXXXX_1.h5'        -> 1
    'task_story_math_XXXXX_1.h5'   -> 2
    'task_working_memory_XXXXX.h5' -> 3
    """
    name = filepath.name
    for idx, task in enumerate(LABELS):
        if name.startswith(task):
            return idx
    raise ValueError(f"Invalid .h5 name: {filepath.name}")


# ---------------------------------------------------------------------------
# 2. File I/O
# ---------------------------------------------------------------------------

def get_dataset_name(filepath: Path) -> str:
    parts = filepath.stem.split("_")[:-1]
    return "_".join(parts)


def load_h5_file(filepath: Path) -> np.ndarray:
    name = get_dataset_name(filepath)
    with h5py.File(filepath, "r") as f:
        data = f.get(name)[()]
    return data


# ---------------------------------------------------------------------------
# 3. Downsample
# ---------------------------------------------------------------------------

def downsample(data: np.ndarray, factor: int = DOWNSAMPLE_FACTOR) -> np.ndarray:
    """
    Decimate along the time axis using a FIR low-pass filter before
    subsampling. Applies a zero-phase FIR anti-alias filter first,
    then subsamples, preserving the brain-relevant frequency bands intact.
    """
    return scipy_decimate(data, factor, ftype="fir", axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# 4. Normalisatie
# ---------------------------------------------------------------------------

# Datatype voor de bevroren statistieken die door de hele pipeline worden
# doorgegeven.  Beide arrays hebben shape (N_CHANNELS, 1) = (248, 1).
ChannelStats = tuple[np.ndarray, np.ndarray]   # (mediaan, IQR)


def compute_channel_stats(filepaths: list[Path]) -> ChannelStats:
    """
    Bereken per-kanaal mediaan en IQR over ALLE opgegeven bestanden samen.

    Alle bestanden worden ingeladen, gedownsampled en samengevoegd langs de
    tijdas voordat de statistieken worden berekend.  Zo reflecteren ze de
    volledige trainingsdistributie, inclusief amplitude-verschillen tussen
    taken en subjects.

    Returns
    -------
    median : np.ndarray, shape (248, 1)
    iqr    : np.ndarray, shape (248, 1)   (= q75 - q25, minimaal 1e-8)
    """
    print(f"\n-- Computing global channel statistics over {len(filepaths)} files…")
    all_data: list[np.ndarray] = []
    for fp in filepaths:
        data = load_h5_file(fp)
        data = downsample(data)          # (248, T_downsampled)
        all_data.append(data)

    # Plak alle tijdstappen aan elkaar: (248, totaal_T)
    combined = np.concatenate(all_data, axis=1)

    median = np.median(combined, axis=1, keepdims=True)           # (248, 1)
    q75    = np.percentile(combined, 75, axis=1, keepdims=True)   # (248, 1)
    q25    = np.percentile(combined, 25, axis=1, keepdims=True)   # (248, 1)
    iqr    = q75 - q25 + 1e-8                                     # (248, 1)

    print(f"   median range : [{median.min():.3e}, {median.max():.3e}]")
    print(f"   IQR   range  : [{iqr.min():.3e},  {iqr.max():.3e}]")
    return median.astype(np.float32), iqr.astype(np.float32)


def zscore_global(data: np.ndarray, median: np.ndarray, iqr: np.ndarray) -> np.ndarray:
    """
    Pas de bevroren globale statistieken toe op één bestand.
    data   : (248, T)
    median : (248, 1)
    iqr    : (248, 1)
    """
    return ((data - median) / iqr).astype(np.float32)


# Bewaar de oude per-bestand functies voor volledigheid / vergelijking,
# maar ze worden niet meer aangeroepen in de hoofdpipeline.

def _zscore_per_file(data: np.ndarray) -> np.ndarray:
    """[LEGACY] Robust z-score per kanaal, berekend per bestand.
    VERNIETIGT amplitude-informatie tussen taken — niet meer gebruiken."""
    median = np.median(data, axis=1, keepdims=True)
    q75    = np.percentile(data, 75, axis=1, keepdims=True)
    q25    = np.percentile(data, 25, axis=1, keepdims=True)
    return ((data - median) / (q75 - q25 + 1e-8)).astype(np.float32)


def _minmax_per_file(data: np.ndarray) -> np.ndarray:
    """[LEGACY] Min-max per kanaal, berekend per bestand."""
    mn = data.min(axis=1, keepdims=True)
    mx = data.max(axis=1, keepdims=True)
    return ((data - mn) / (mx - mn + 1e-8)).astype(np.float32)


# ---------------------------------------------------------------------------
# 5. Sliding window
# ---------------------------------------------------------------------------

def make_windows(
        data:   np.ndarray,
        label:  int,
        window: int = WINDOW_SIZE,
        stride: int = WINDOW_STRIDE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Snijd het signaal in overlappende windows.
    X : (N, 248, window)
    y : (N,)
    """
    _, T   = data.shape
    starts = range(0, T - window + 1, stride)
    X = np.stack([data[:, s: s + window] for s in starts])
    y = np.full(len(X), label, dtype=np.int64)
    return X, y


# ---------------------------------------------------------------------------
# 6. Verwerk één / meerdere bestanden
# ---------------------------------------------------------------------------

def process_file(
        filepath: Path,
        stats:    ChannelStats,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Laad, downsample, normaliseer (met globale stats) en maak windows.

    Parameters
    ----------
    filepath : pad naar een .h5 bestand
    stats    : (mediaan, IQR) berekend over de trainingsset — zie
               compute_channel_stats()
    """
    median, iqr = stats
    label = get_label(filepath)
    data  = load_h5_file(filepath)
    data  = downsample(data)
    data  = zscore_global(data, median, iqr)   # ← globale stats
    X, y  = make_windows(data, label)
    return X, y


def process_files(
        filepaths: list[Path],
        stats:     ChannelStats,
        verbose:   bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Verwerk meerdere bestanden en concateneer de windows.

    Parameters
    ----------
    stats : doorgeef de globale (mediaan, IQR) van de trainingsset
    """
    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []

    for fp in filepaths:
        X, y = process_file(fp, stats)
        all_X.append(X)
        all_y.append(y)
        if verbose:
            label_name = LABELS[y[0]]
            print(f"Loaded: {fp.name:45s} with {len(y):4d} windows [{label_name}]")

    return np.concatenate(all_X), np.concatenate(all_y)


# ---------------------------------------------------------------------------
# 7. Dataset / DataLoader hulpfuncties
# ---------------------------------------------------------------------------

class MEGDataset(Dataset):
    """Wrapper om numpy-arrays in een PyTorch Dataset te stoppen."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)   # float32 tensor
        self.y = torch.from_numpy(y)   # int64  tensor

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_files(folder: Path) -> list[Path]:
    """Geef alle .h5 bestanden in een map, gesorteerd voor reproduceerbaarheid."""
    files = sorted(folder.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {folder}")
    return files


def make_loader(ds, shuffle: bool, batch_size: int = BATCH_SIZE) -> DataLoader:
    return DataLoader(
        ds,
        batch_size         = batch_size,
        shuffle            = shuffle,
        num_workers        = NUM_WORKERS,
        pin_memory         = True,
        drop_last          = shuffle,
        worker_init_fn     = _worker_init,
        persistent_workers = NUM_WORKERS > 0,
    )


# ---------------------------------------------------------------------------
# 8. Intra-subject loaders
# ---------------------------------------------------------------------------

def build_loaders(
        train_folder: Path,
        test_folder:  Path,
        verbose:      bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Bouw train / val / test DataLoaders voor het Intra-experiment.

    Stappen:
      1. Bereken globale stats over ALLEEN de trainingsbestanden.
      2. Verwerk trein- en testbestanden met die bevroren stats.
      3. Splits train in train + val.
    """
    if verbose:
        print(f"\n--Loading *training* files from: {train_folder}")
    train_files = get_files(train_folder)

    # Stap 1: globale statistieken op trainingsdata
    stats = compute_channel_stats(train_files)

    # Stap 2: verwerk met bevroren stats
    X_train, y_train = process_files(train_files, stats, verbose)

    if verbose:
        print(f"\n--Loading *testing* files from: {test_folder}")
    test_files   = get_files(test_folder)
    X_test, y_test = process_files(test_files, stats, verbose)   # zelfde stats!

    # Stap 3: train/val split
    full_train = MEGDataset(X_train, y_train)
    n_val      = int(len(full_train) * VAL_SPLIT)
    n_train    = len(full_train) - n_val
    gen        = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(full_train, [n_train, n_val], generator=gen)
    test_ds    = MEGDataset(X_test, y_test)

    return (
        make_loader(train_ds, shuffle=True),
        make_loader(val_ds,   shuffle=False),
        make_loader(test_ds,  shuffle=False),
    )


# ---------------------------------------------------------------------------
# 9. Cross-subject loaders (chunked)
# ---------------------------------------------------------------------------

def build_loaders_chunked(
        train_folder:    Path,
        test_folders:    list[Path],
        files_per_chunk: int  = FILES_PER_CHUNK,
        verbose:         bool = True,
) -> tuple[list[list[Path]], list[Path], ChannelStats]:
    """
    Voor het Cross-experiment: splits de trainingsbestanden in chunks (hint 3)
    en bereken globale statistieken over alle trainingsbestanden.

    Returns
    -------
    chunks     : lijst van bestandslijsten
    all_test   : gecombineerde testbestandslijst
    stats      : (mediaan, IQR) — doorgeef aan process_files() en
                 build_val_loader_from_chunks()
    """
    all_train = get_files(train_folder)
    chunks: list[list[Path]] = [
        all_train[i: i + files_per_chunk]
        for i in range(0, len(all_train), files_per_chunk)
    ]

    all_test: list[Path] = []
    for folder in test_folders:
        if folder.exists():
            all_test.extend(get_files(folder))

    if verbose:
        print(f"--Cross train: {len(all_train)} files in "
              f"{len(chunks)} chunks of {files_per_chunk}")
        print(f"--Cross test : {len(all_test)} files across "
              f"{len(test_folders)} test folders")

    # Globale statistieken over alle trainingsbestanden
    stats = compute_channel_stats(all_train)

    return chunks, all_test, stats   # ← stats is nieuw t.o.v. de oude versie


def build_val_loader_from_chunks(
        file_chunks:  list[list[Path]],
        stats:        ChannelStats,
        val_fraction: float = VAL_SPLIT,
) -> tuple[DataLoader, np.ndarray, np.ndarray]:
    """
    Bouw een validatie-DataLoader door val_fraction van elke chunk te samplen.

    Parameters
    ----------
    stats : bevroren (mediaan, IQR) van de trainingsset
    """
    val_indices: list[int]      = []
    all_X_list:  list[np.ndarray] = []
    all_y_list:  list[np.ndarray] = []
    offset = 0

    rng = np.random.default_rng(SEED)

    for chunk in file_chunks:
        X, y  = process_files(chunk, stats, verbose=False)
        n     = len(y)
        n_val = max(1, int(n * val_fraction))
        chosen = rng.choice(n, size=n_val, replace=False)
        val_indices.extend((chosen + offset).tolist())
        all_X_list.append(X)
        all_y_list.append(y)
        offset += n

    X_all    = np.concatenate(all_X_list)
    y_all    = np.concatenate(all_y_list)
    full_ds  = MEGDataset(X_all, y_all)
    val_ds   = Subset(full_ds, val_indices)

    return make_loader(val_ds, shuffle=False), X_all, y_all