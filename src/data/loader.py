"""
Reads the .h5 files, downsamples (hint 2) and normalises (hint 1)
Cuts the signal into short windows (each window = one training sample)

Normalisation strategy:
    Statistics (median + IQR) are computed over ALL training files combined,
    then frozen and applied to train/val/test. This preserves the amplitude
    differences between tasks (rest vs. motor vs. math vs. memory), which
    are discriminative features for the model.
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

# 1. Labels
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

# 2. Read .h5 files
def get_dataset_name(filepath: Path) -> str:
    parts = filepath.stem.split("_")[:-1]
    return "_".join(parts)

def load_h5_file(filepath: Path) -> np.ndarray:
    name = get_dataset_name(filepath)
    with h5py.File(filepath, "r") as f:
        data = f.get(name)[()]
    return data

# 3. Downsample
def downsample(data: np.ndarray, factor: int = DOWNSAMPLE_FACTOR) -> np.ndarray:
    """
    Decimate along the time axis using a FIR low-pass filter before
    subsampling. Applies a zero-phase FIR anti-alias filter first,
    then subsamples, preserving the brain-relevant frequency bands intact.
    """
    return scipy_decimate(data, factor, ftype = "fir", axis = 1).astype(np.float32)

# 4. Normalisation
# Type alias for the frozen statistics passed through the pipeline.
# Both arrays have shape (N_CHANNELS, 1) = (248, 1).
ChannelStats = tuple[np.ndarray, np.ndarray]   # (median, IQR)
def compute_channel_stats(filepaths: list[Path]) -> ChannelStats:
    """
    Compute per-channel median and IQR over ALL given files combined.

    All files are loaded, downsampled and concatenated along the time axis
    before computing statistics. This way they reflect the full training
    distribution, including amplitude differences between tasks and subjects.
    """
    print(f"\n-- Computing global channel statistics over {len(filepaths)} files...")
    all_data: list[np.ndarray] = []
    for fp in filepaths:
        data = load_h5_file(fp)
        data = downsample(data)          # (248, T_downsampled)
        all_data.append(data)

    # Concatenate all time steps: (248, total_T)
    combined = np.concatenate(all_data, axis = 1)

    median = np.median(combined, axis = 1, keepdims = True)           # (248, 1)
    q75    = np.percentile(combined, 75, axis = 1, keepdims = True)   # (248, 1)
    q25    = np.percentile(combined, 25, axis = 1, keepdims = True)   # (248, 1)
    iqr    = q75 - q25 + 1e-8                                         # (248, 1)

    print(f"   median range : [{median.min():.3e}, {median.max():.3e}]")
    print(f"   IQR   range  : [{iqr.min():.3e},  {iqr.max():.3e}]")
    return median.astype(np.float32), iqr.astype(np.float32)

# 4a using zscore
def zscore_global(data: np.ndarray, median: np.ndarray, iqr: np.ndarray) -> np.ndarray:
    """
    Apply the frozen global statistics to a single file.
    data   : (248, T)
    median : (248, 1)
    iqr    : (248, 1)
    """
    return ((data - median) / iqr).astype(np.float32)
# 4b using minmax per file
def minmax_per_file(data: np.ndarray) -> np.ndarray:
    """ Min-max per channel, computed per file."""
    mn = data.min(axis = 1, keepdims = True)
    mx = data.max(axis = 1, keepdims = True)
    return ((data - mn) / (mx - mn + 1e-8)).astype(np.float32)

def normalization(data: np.ndarray, median, iqr, method: str = NORMALIZATION) -> np.ndarray:
    if   method == "zscore": return zscore_global(data, median = median, iqr = iqr)
    elif method == "minmax": return minmax_per_file(data)
    print("--WARNINg: No (correct) normalization set, using 'zscore' -- ")
    return zscore_global(data, median, iqr)

# 5. Sliding window
def make_windows(
        data:   np.ndarray,
        label:  int,
        window: int = WINDOW_SIZE,
        stride: int = WINDOW_STRIDE
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Cut the signal into overlapping windows.
    X : (N, 248, window)
    y : (N,)
    """
    _, T   = data.shape
    starts = range(0, T - window + 1, stride)
    X      = np.stack([data[:, s : s + window] for s in starts])
    y      = np.full(len(X), label, dtype = np.int64)
    return X, y

# 6. Process one / multiple files
def process_file(
        filepath: Path,
        stats:    ChannelStats
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Load, downsample, normalise (with global stats) and create windows
    """
    median, iqr = stats
    label       = get_label(filepath)
    data        = load_h5_file(filepath)
    data        = downsample(data)
    data        = normalization(data, median, iqr)
    X, y        = make_windows(data, label)
    return X, y

def process_files(
        filepaths: list[Path],
        stats:     ChannelStats,
        verbose:   bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Process multiple files and concatenate their windows.
    Pass the global (median, IQR) from the training set via stats.
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

# 7. Dataset / DataLoader helpers
class MEGDataset(Dataset):
    """Wraps numpy arrays into a PyTorch Dataset"""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)   # float32 tensor
        self.y = torch.from_numpy(y)   # int64  tensor

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_files(folder: Path) -> list[Path]:
    """Return all .h5 files in a folder, sorted for reproducibility."""
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
        persistent_workers = NUM_WORKERS > 0)

# 8. Intra-subject loaders
def build_loaders(
        train_folder: Path,
        test_folder:  Path,
        verbose:      bool = True
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders for the Intra experiment.

    Steps:
      1. Compute global stats over ONLY the training files.
      2. Process train and test files with those frozen stats.
      3. Split train into train + val.
    """
    if verbose:
        print(f"\n--Loading *training* files from: {train_folder}")
    train_files = get_files(train_folder)

    # Step 1: global statistics on training data only
    stats = compute_channel_stats(train_files)

    # Step 2: process with frozen stats
    X_train, y_train = process_files(train_files, stats, verbose)

    if verbose:
        print(f"\n--Loading *testing* files from: {test_folder}")
    test_files     = get_files(test_folder)
    X_test, y_test = process_files(test_files, stats, verbose)   # same stats!

    # Step 3: train / val split
    full_train       = MEGDataset(X_train, y_train)
    n_val            = int(len(full_train) * VAL_SPLIT)
    n_train          = len(full_train) - n_val
    gen              = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(full_train, [n_train, n_val], generator = gen)
    test_ds          = MEGDataset(X_test, y_test)

    return (
        make_loader(train_ds, shuffle = True),
        make_loader(val_ds,   shuffle = False),
        make_loader(test_ds,  shuffle = False))

# 9. Cross-subject loaders (chunked)
def build_loaders_chunked(
        train_folder:    Path,
        test_folders:    list[Path],
        files_per_chunk: int  = FILES_PER_CHUNK,
        verbose:         bool = True
    ) -> tuple[list[list[Path]], list[Path], ChannelStats]:
    """
    For the Cross experiment: split training files into chunks (hint 3)
    and compute global statistics over all training files.
    """
    all_train = get_files(train_folder)
    chunks: list[list[Path]] = [
        all_train[i : i + files_per_chunk]
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

    # Global statistics over all training files
    stats = compute_channel_stats(all_train)

    return chunks, all_test, stats

def build_val_loader_from_chunks(
        file_chunks:  list[list[Path]],
        stats:        ChannelStats,
        val_fraction: float = VAL_SPLIT
    ) -> tuple[DataLoader, np.ndarray, np.ndarray]:
    """
    Build a validation DataLoader by sampling val_fraction of windows
    from every training chunk.
    """
    val_indices: list[int]        = []
    all_X_list:  list[np.ndarray] = []
    all_y_list:  list[np.ndarray] = []
    offset       = 0

    rng = np.random.default_rng(SEED)

    for chunk in file_chunks:
        X, y   = process_files(chunk, stats, verbose = False)
        n      = len(y)
        n_val  = max(1, int(n * val_fraction))
        chosen = rng.choice(n, size = n_val, replace = False)
        val_indices.extend((chosen + offset).tolist())
        all_X_list.append(X)
        all_y_list.append(y)
        offset += n

    X_all   = np.concatenate(all_X_list)
    y_all   = np.concatenate(all_y_list)
    full_ds = MEGDataset(X_all, y_all)
    val_ds  = Subset(full_ds, val_indices)

    return make_loader(val_ds, shuffle = False), X_all, y_all
