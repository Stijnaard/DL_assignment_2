"""
Reads the .h5 files, downsamples (hint 2) and normalises (hint 1)
Cuts the signal into short windows (each window = one training sample)
"""

from pathlib import Path
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from scipy.signal import decimate as scipy_decimate
 
from src.config.config import *

# Helper
def _worker_init(worker_id: int):
    """Give each DataLoader worker its own reproducible seed"""
    np.random.seed(SEED + worker_id)

# 1. Assign labeles to the .h5 data
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

# 2. Read a .h5 files
def get_dataset_name(filepath: Path) -> str:
    parts = filepath.stem.split("_")[:-1]
    return "_".join(parts)

def load_h5_file(filepath: Path) -> np.ndarray:
    name = get_dataset_name(filepath)
    with h5py.File(filepath, "r") as f:
        data = f.get(name)[()]
    return data

# 3. Downsample data (hint 2)
def downsample(data: np.ndarray, factor: int = DOWNSAMPLE_FACTOR) -> np.ndarray:
    """
    Decimate along the time axis using a FIR low-pass filter before
    subsampling. Applies a zero-phase FIR anti-alias filter first,
    then subsamples, preserving the brain-relevant frequency bands intact.
    """
    # Decimate operates along axis = 1 (time), returns float64 -> cast to float32
    return scipy_decimate(data, factor, ftype = "fir", axis = 1).astype(np.float32)

# 4a. Normalisation — standard z-score (Intra)
def zscore(data: np.ndarray) -> np.ndarray:
    """Standard z-score per channel: (x - mean) / std.
    Best for Intra: single subject, clean signal, no inter-subject artefacts."""
    mean = data.mean(axis=1, keepdims=True)
    std  = data.std( axis=1, keepdims=True)
    return ((data - mean) / (std + 1e-8)).astype(np.float32)

# 4b. Normalisation — robust z-score (Cross)
def zscore_robust(data: np.ndarray) -> np.ndarray:
    """Robust z-score per channel: (x - median) / IQR.
    Best for Cross: multiple subjects with varying artefact levels; median/IQR
    resist spike inflation that would compress the real neural signal."""
    median = np.median(data, axis=1, keepdims=True)
    q75    = np.percentile(data, 75, axis=1, keepdims=True)
    q25    = np.percentile(data, 25, axis=1, keepdims=True)
    return ((data - median) / (q75 - q25 + 1e-8)).astype(np.float32)

# 4c. Normalisation — min-max scaling
def minmax(data: np.ndarray) -> np.ndarray:
    """Scales each sensor to [0, 1] range"""
    mn = data.min(axis=1, keepdims=True)
    mx = data.max(axis=1, keepdims=True)
    return ((data - mn) / (mx - mn + 1e-8)).astype(np.float32)

# 4. Normalize the data
def normalize(data: np.ndarray, method: str = NORMALIZATION,
              robust: bool = False) -> np.ndarray:
    """Apply the chosen normalisation method.
    robust=True forces median/IQR z-score regardless of method setting."""
    if method == "minmax": return minmax(data)
    if robust:             return zscore_robust(data)
    return zscore(data)

# 5. Sliding window
def make_windows(
        data: np.ndarray,
        label: int,
        window: int = WINDOW_SIZE,
        stride: int = WINDOW_STRIDE,
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Each window becomes one training sample.
    With window = 200, stride = 100: 3562-step recording -> +/- 35 samples.
    X : (N, 248, window) N windows, each (sensors x time)
    y : (N,)             Same label repeated N times
    """
    _, T = data.shape
    starts = range(0, T - window + 1, stride)
    X = np.stack([data[:, s : s + window] for s in starts])
    y = np.full(len(X), label, dtype = np.int64)
    return X, y

# 6. Process a single .h5 file
def process_file(filepath: Path, robust: bool = False) -> tuple[np.ndarray, np.ndarray]:
    label = get_label(filepath)
    data  = load_h5_file(filepath)
    data  = downsample(data)
    data  = normalize(data, robust=robust)
    X, y  = make_windows(data, label)
    return X, y

def process_files(filepaths: list[Path], verbose: bool = True,
                  robust: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Process multiple files and concatenate their windows
    (These are the chunks during Cross training (hint 3))
    """
    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []

    for fp in filepaths:
        X, y = process_file(fp, robust=robust)
        all_X.append(X)
        all_y.append(y)
        if verbose:
            label_name = LABELS[y[0]]
            print(f"Loaded: {fp.name:45s} with {len(y):4d} windows [{label_name}]")

    return np.concatenate(all_X), np.concatenate(all_y)

# Create the dataset
class MEGDataset(Dataset):
    """MEGDataset used to store data"""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X) # Pytorch float32 tensor
        self.y = torch.from_numpy(y) # Pytorch int64 tensor

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Build the DataLoaders from .h5 files
def get_files(folder: Path) -> list[Path]:
    """Return all .h5 files in a folder"""
    files = sorted(folder.glob("*.h5")) # Sorted for reproducability
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {folder}")
    return files

def make_loader(ds, shuffle: bool, batch_size: int = BATCH_SIZE) -> DataLoader:
    return DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = NUM_WORKERS,
        pin_memory  = True,                     # Faster GPU handling
        drop_last  = shuffle,                   # Drop incomplete last batch only during training
        worker_init_fn = _worker_init,          # Reproducible workers
        persistent_workers = NUM_WORKERS > 0
    ) 
    
def build_loaders(
        train_folder: Path,
        test_folder: Path,
        verbose: bool = True,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load all files from train and test folders, build PyTorch DataLoaders
    """
    if verbose:
        print(f"\n--Loading *training* files from: {train_folder}")
    train_files      = get_files(train_folder)
    X_train, y_train = process_files(train_files, verbose)

    if verbose:
        print(f"\n--Loading *testing* files from: {test_folder}")
    test_files     = get_files(test_folder)
    X_test, y_test = process_files(test_files, verbose)

    # Split training data into train + validation
    full_train = MEGDataset(X_train, y_train)
    n_val      = int(len(full_train) * VAL_SPLIT)
    n_train    = len(full_train) - n_val
    gen        = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(full_train, [n_train, n_val], generator = gen)
    test_ds    = MEGDataset(X_test, y_test)

    return (make_loader(train_ds, shuffle = True),
        make_loader(val_ds, shuffle = False),
        make_loader(test_ds, shuffle = False))

def build_loaders_chunked(
        train_folder: Path,
        test_folders: list[Path],
        files_per_chunk: int = FILES_PER_CHUNK,
        verbose: bool = True,
    ) -> tuple[list[list[Path]], list[Path]]:
    """
    For Cross training: instead of loading all files at once,
    return the file lists split into chunks (hint 3).
    """
    all_train = get_files(train_folder)
    chunks: list[list[Path]]
    chunks = [all_train[i : i + files_per_chunk]
        for i in range(0, len(all_train), files_per_chunk)]

    all_test: list[Path] = []
    for folder in test_folders:
        if folder.exists():
            all_test.extend(get_files(folder))

    if verbose:
        print(f"--Cross train: {len(all_train)} files in {len(chunks)} chunks of {files_per_chunk}")
        print(f"--Cross test : {len(all_test)} files across {len(test_folders)} test folders")

    return chunks, all_test

def build_val_loader_from_chunks(
        file_chunks: list[list[Path]],
        val_fraction: float = VAL_SPLIT,
    ) -> tuple[DataLoader, np.ndarray, np.ndarray]:
    """
    Build a validation DataLoader by sampling 'val_fraction' of windows
    from every training chunk
    """
    val_indices: list[int] = []
    all_X_list:  list[np.ndarray] = []
    all_y_list:  list[np.ndarray] = []
    offset = 0
 
    rng = np.random.default_rng(SEED)
 
    for chunk in file_chunks:
        X, y = process_files(chunk, verbose=False, robust=True)
        n     = len(y)
        n_val = max(1, int(n * val_fraction))
        chosen = rng.choice(n, size = n_val, replace = False)
        val_indices.extend((chosen + offset).tolist())
        all_X_list.append(X)
        all_y_list.append(y)
        offset += n
 
    X_all = np.concatenate(all_X_list)
    y_all = np.concatenate(all_y_list)
    full_ds  = MEGDataset(X_all, y_all)
    val_ds   = Subset(full_ds, val_indices)
 
    return make_loader(val_ds, shuffle = False), X_all, y_all