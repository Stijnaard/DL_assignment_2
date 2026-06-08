"""
Reads the .h5 files, downsamples (hint 2) and normalises (hint 1)
Cuts the signal into short windows (each window = one training sample)

Normalisation strategy:
  - Intra: statistics (median + IQR) are computed over all training files
    from the single subject, then frozen and applied to train/val/test.
    This preserves inter-task amplitude differences for that subject.

  - Cross: statistics are computed per subject (identified by the 6-digit ID
    in the filename). Each subject's files are normalised with that subject's
    own frozen stats. Test subjects are unseen, so they are normalised with
    their own per-subject stats computed on-the-fly from their own files.
    This avoids the pooling problem where two subjects with very different
    baseline amplitudes produce stats that fit neither well.
"""

from pathlib import Path
import re
import h5py
import numpy as np

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


def get_subject_id(filepath: Path) -> str:
    """
    Extract the 6-digit subject identifier from a filename.
    Example: 'task_motor_105923_1.h5' -> '105923'
    """
    match = re.search(r"_(\d{6})_", filepath.name)
    if match:
        return match.group(1)
    raise ValueError(f"No 6-digit subject ID found in: {filepath.name}")


# 2. File I/O

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
ChannelStats    = tuple[np.ndarray, np.ndarray]          # (median, IQR)
SubjectStatsMap = dict[str, ChannelStats]                # subject_id -> stats


def compute_channel_stats(filepaths: list[Path]) -> ChannelStats:
    """
    Compute per-channel median and IQR over all given files combined.
    Used for Intra (single subject) and as a building block for per-subject
    stats in Cross.

    Returns
    -------
    median : np.ndarray, shape (248, 1)
    iqr    : np.ndarray, shape (248, 1)   (= q75 - q25, minimum 1e-8)
    """
    all_data: list[np.ndarray] = []
    for fp in filepaths:
        data = load_h5_file(fp)
        data = downsample(data)
        all_data.append(data)

    combined = np.concatenate(all_data, axis = 1)   # (248, total_T)
    median   = np.median(combined, axis = 1, keepdims = True)
    q75      = np.percentile(combined, 75, axis = 1, keepdims = True)
    q25      = np.percentile(combined, 25, axis = 1, keepdims = True)
    iqr      = q75 - q25 + 1e-8
    return median.astype(np.float32), iqr.astype(np.float32)


def compute_subject_stats(filepaths: list[Path], verbose: bool = True) -> SubjectStatsMap:
    """
    Group files by subject ID and compute per-subject channel statistics.

    For Cross training: the 64 training files come from 2 subjects. Pooling
    them into a single set of statistics produces a poor fit for both because
    subjects differ in overall amplitude. Computing one set of stats per
    subject and applying them file-by-file keeps the inter-task amplitude
    information intact while removing subject-level baseline differences.

    Returns
    -------
    subject_stats : dict mapping subject_id (str) -> (median, IQR)
    """
    # Group files by subject
    groups: dict[str, list[Path]] = {}
    for fp in filepaths:
        sid = get_subject_id(fp)
        groups.setdefault(sid, []).append(fp)

    if verbose:
        print(f"\n-- Computing per-subject channel statistics "
              f"({len(groups)} subjects, {len(filepaths)} files total)...")

    subject_stats: SubjectStatsMap = {}
    for sid, fps in sorted(groups.items()):
        stats = compute_channel_stats(fps)
        subject_stats[sid] = stats
        median, iqr = stats
        if verbose:
            print(f"   Subject {sid}: {len(fps):2d} files | "
                  f"median range [{median.min():.3e}, {median.max():.3e}] | "
                  f"IQR range [{iqr.min():.3e}, {iqr.max():.3e}]")

    return subject_stats


def zscore_global(data: np.ndarray, median: np.ndarray, iqr: np.ndarray) -> np.ndarray:
    """
    Apply frozen statistics to a single recording.
    data   : (248, T)
    median : (248, 1)
    iqr    : (248, 1)
    """
    return ((data - median) / iqr).astype(np.float32)


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
        filepath:      Path,
        stats:         ChannelStats | None      = None,
        subject_stats: SubjectStatsMap | None   = None
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Load, downsample, normalise and create windows for one file.

    Normalisation priority:
      1. subject_stats (Cross): look up this file's subject ID and use
         the matching frozen stats.
      2. stats (Intra): apply a single shared set of frozen stats.
      Both preserve inter-task amplitude differences within a subject.
    """
    label = get_label(filepath)
    data  = load_h5_file(filepath)
    data  = downsample(data)

    if subject_stats is not None:
        sid            = get_subject_id(filepath)
        # Test subjects are unseen — fall back to their own on-the-fly stats
        if sid not in subject_stats:
            fallback_stats = compute_channel_stats([filepath])
            subject_stats[sid] = fallback_stats
        median, iqr = subject_stats[sid]
    elif stats is not None:
        median, iqr = stats
    else:
        raise ValueError("process_file: provide either stats or subject_stats")

    data   = zscore_global(data, median, iqr)
    X, y   = make_windows(data, label)
    return X, y


def process_files(
        filepaths:     list[Path],
        stats:         ChannelStats | None     = None,
        subject_stats: SubjectStatsMap | None  = None,
        verbose:       bool                    = True
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Process multiple files and concatenate their windows.

    Pass either:
      - stats         for Intra (single shared stats)
      - subject_stats for Cross (per-subject stats map)
    """
    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []

    for fp in filepaths:
        X, y = process_file(fp, stats = stats, subject_stats = subject_stats)
        all_X.append(X)
        all_y.append(y)
        if verbose:
            label_name = LABELS[y[0]]
            print(f"Loaded: {fp.name:45s} with {len(y):4d} windows [{label_name}]")

    return np.concatenate(all_X), np.concatenate(all_y)


# 7. Dataset / DataLoader helpers

class MEGDataset(Dataset):
    """Wraps numpy arrays into a PyTorch Dataset."""

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
      1. Compute global stats over ONLY the training files (one subject).
      2. Process train and test files with those frozen stats.
      3. Split train into train + val.
    """
    if verbose:
        print(f"\n--Loading *training* files from: {train_folder}")
    train_files = get_files(train_folder)

    # Step 1: global statistics on training data only (single subject)
    print(f"\n-- Computing global channel statistics over {len(train_files)} files...")
    stats          = compute_channel_stats(train_files)
    median, iqr    = stats
    print(f"   median range : [{median.min():.3e}, {median.max():.3e}]")
    print(f"   IQR   range  : [{iqr.min():.3e},  {iqr.max():.3e}]")

    # Step 2: process with frozen stats
    X_train, y_train = process_files(train_files, stats = stats, verbose = verbose)

    if verbose:
        print(f"\n--Loading *testing* files from: {test_folder}")
    test_files     = get_files(test_folder)
    X_test, y_test = process_files(test_files, stats = stats, verbose = verbose)

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
    ) -> tuple[list[list[Path]], list[Path], SubjectStatsMap]:
    """
    For the Cross experiment: split training files into chunks (hint 3)
    and compute per-subject statistics over all training files.

    Returns
    -------
    chunks        : list of file lists for chunked training
    all_test      : combined test file list
    subject_stats : per-subject (median, IQR) map — pass to process_files()
                    and build_val_loader_from_chunks()
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

    # Per-subject statistics over all training files
    subject_stats = compute_subject_stats(all_train, verbose = verbose)

    return chunks, all_test, subject_stats


def build_val_loader_from_chunks(
        file_chunks:   list[list[Path]],
        subject_stats: SubjectStatsMap,
        val_fraction:  float = VAL_SPLIT
    ) -> tuple[DataLoader, np.ndarray, np.ndarray]:
    """
    Build a validation DataLoader by sampling val_fraction of windows
    from every training chunk, using per-subject normalisation.
    """
    val_indices: list[int]        = []
    all_X_list:  list[np.ndarray] = []
    all_y_list:  list[np.ndarray] = []
    offset       = 0

    rng = np.random.default_rng(SEED)

    for chunk in file_chunks:
        X, y   = process_files(chunk, subject_stats = subject_stats, verbose = False)
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