"""
Reads the .h5 files, downsamples (hint 2) and normalises (hint 1)
Cuts the signal into short windows (each window = one training sample)
"""

from pathlib import Path
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from src.config.config import *
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
    """Keep every 'n'-th time sample, to reduce training time"""
    return data[:, ::factor] # shape: (248, T // factor)

# 4a. Normalisation (hint 3)
def zscore(data: np.ndarray) -> np.ndarray:
    """
    Z-score normalisation applied PER CHANNEL (each row independently)
    normalised = (value - mean) / std
    """
    mean = data.mean(axis = 1, keepdims = True)
    std  = data.std( axis = 1, keepdims = True)
    return ((data - mean) / (std + 1e-8)).astype(np.float32)
# 4b. Normalisation using min-max scaling (hint 3)
def minmax(data: np.ndarray) -> np.ndarray:
    """Scales each sensor to [0, 1] range"""
    mn = data.min(axis = 1, keepdims = True)
    mx = data.max(axis = 1, keepdims = True)
    return ((data - mn) / (mx - mn + 1e-8)).astype(np.float32)

# 4. Normalize the data
def normalize(data: np.ndarray, method: str = NORMALIZATION) -> np.ndarray:
    """Apply the chosen normalisation method"""
    if method == "minmax": return minmax(data)
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
    y = np.full(len(X), label, dtype=np.int64)
    return X, y

# 6. Process a single .h5 file
def process_file(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    label = get_label(filepath)
    data  = load_h5_file(filepath)    # (248, T)
    data  = downsample(data)          # (248, T//10)
    data  = normalize(data)           # (248, T//10)
    X, y  = make_windows(data, label) # (N, 248, W), (N,)
    return X, y

def process_files(filepaths: list[Path], verbose: bool = True
        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Process multiple files and concatenate their windows
    (These are the chunks during Cross training (hint 3))
    """
    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []

    for fp in filepaths:
        X, y = process_file(fp)
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
    """Return all .h5 files in a folder, sorted"""
    files = sorted(folder.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {folder}")
    return files

def build_loaders(
        train_folder: Path,
        test_folder: Path,
        batch_size: int = BATCH_SIZE,
        verbose: bool = True,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load all files from train and test folders, build PyTorch DataLoaders
    """
    if verbose:
        print(f"\n--Loading *training* files from: {train_folder}")
    train_files = get_files(train_folder)
    X_train, y_train = process_files(train_files, verbose)

    if verbose:
        print(f"\n--Loading *testing* files from: {test_folder}")
    test_files = get_files(test_folder)
    X_test, y_test = process_files(test_files, verbose)

    # Split training data into train + validation
    full_train = MEGDataset(X_train, y_train)
    n_val   = int(len(full_train) * VAL_SPLIT)
    n_train = len(full_train) - n_val
    gen = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(full_train, [n_train, n_val], generator = gen)
    test_ds = MEGDataset(X_test, y_test)

    def make_loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = shuffle,
            num_workers = NUM_WORKERS,
            pin_memory  = True,   # Faster GPU handling
            drop_last  = shuffle) # drop incomplete last batch only during training

    return (make_loader(train_ds, True),
        make_loader(val_ds, False),
        make_loader(test_ds, False))

def build_loaders_chunked(
        train_folder: Path,
        test_folders: list[Path],
        files_per_chunk: int = 8,
        verbose: bool = True,
    ) -> tuple[list[list[Path]], list[Path]]:
    """
    For Cross training: instead of loading all files at once,
    return the file lists split into chunks (hint 3).
    """
    all_train = get_files(train_folder)
    chunks: list[list[Path]] = [all_train[i : i + files_per_chunk]
        for i in range(0, len(all_train), files_per_chunk)]

    all_test: list[Path] = []
    for folder in test_folders:
        if folder.exists():
            all_test.extend(get_files(folder))

    if verbose:
        print(f"--Cross train: {len(all_train)} files in {len(chunks)} chunks of {files_per_chunk}")
        print(f"--Cross test : {len(all_test)} files across {len(test_folders)} test folders")

    return chunks, all_test
