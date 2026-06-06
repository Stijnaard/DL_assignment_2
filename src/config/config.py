from pathlib import Path

# (All) models and their representation
ALL_MODELS = ["lstm", "rnn", "gru", "eegnet", "cnn1d",
    "cnn_transformer", "cnn1d_resnet", "temporal_attention"]
MODEL_DISPLAY = {
    "lstm":               "Bi-dir. stacked LSTM",
    "rnn":                "RNN",
    "gru":                "GRU",
    "eegnet":             "EEGNet",
    "cnn1d":              "CNN-1D",
    "cnn_transformer":    "CNN+Transformer",
    "cnn1d_resnet":       "CNN1D-ResNet",
    "temporal_attention": "Temporal Attention"
}
MODEL = "cnn1d" # Default model if none selected

# Clasification lables
LABELS = ["rest", "task_motor", "task_story_math", "task_working_memory"]
NUM_CLASSES = len(LABELS)

# Paths for datasets
DATA_ROOT = Path("datasets")
INTRA_TRAIN = DATA_ROOT / "Intra" / "train"
INTRA_TEST  = DATA_ROOT / "Intra" / "test"
CROSS_TRAIN = DATA_ROOT / "Cross" / "train"
CROSS_TEST1 = DATA_ROOT / "Cross" / "test1"
CROSS_TEST2 = DATA_ROOT / "Cross" / "test2"
CROSS_TEST3 = DATA_ROOT / "Cross" / "test3"

# Output folders
OUTPUT_DIR      = Path("outputs")
FIGURES_DIR     = OUTPUT_DIR / "figures"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_DIR     = OUTPUT_DIR / "results"

# Data preprocessing
ORIGINAL_FS   = 2034       # The MEG device records at 2034 (Hz)
N_CHANNELS    = 248
DOWNSAMPLE_FACTOR = 50   # Take every 'n' measurement
TARGET_FS     = ORIGINAL_FS // DOWNSAMPLE_FACTOR
WINDOW_SIZE   = 200      # Time steps per sample
WINDOW_STRIDE = 100      # Step between windows (50% overlap)
NORMALIZATION = "zscore" # Hint 1: normalisation

# Training settings
DEVICE        = "cuda" # "cuda" for GPU (recommended), "cpu" as fallback
SEED          = 42     # For reproducibility
EPOCHS        = 50
BATCH_SIZE    = 32
LEARNING_RATE = 5e-4
WEIGHT_DECAY  = 5e-4   # L2 regularisation
GRAD_CLIP     = 1.0    # Clipping gradients --> to prevent exploding
VAL_SPLIT     = 0.20
PATIENCE      = 20
OPTIMIZER     = "adam_w"   # Or "adam"
LABEL_SMOOTHING = 0.10
MIXUP_ALPHA_INTRA = 0.0   # MixUp strength
MIXUP_ALPHA_CROSS = 0.2   # You want Intra to overfitt more than cross

# Hint 3: memory management for Cross
FILES_PER_CHUNK = 8
NUM_WORKERS     = 4        # Parallel workers


# ========================= #
# Model specific parameters # 
# ========================= #
# Stacked bi-directional LSTM
STACKED_HIDDEN_SIZE  = 64
STACKED_NUM_LAYERS   = 2
STACKED_DROPOUT_RATE = 0.2

# Rnn
RNN_HIDDEN  = 128
RNN_LAYERS  = 2    # Stacked layers
RNN_DROPOUT = 0.3
RNN_BIDIR   = True # Bidirectional

# GRU
GRU_HIDDEN  = 256
GRU_LAYERS  = 3
GRU_DROPOUT = 0.3
GRU_BIDIR   = True

# EEGNet
EEGNET_F1          = 8                      # Temporal filters in block 1
EEGNET_D           = 2                      # Depth multiplier (spatial filters = F1 * D)
EEGNET_F2          = EEGNET_F1 * EEGNET_D   # Total spatial filters
EEGNET_KERNEL_SIZE = 64                     # Temporal kernel
EEGNET_DROPOUT     = 0.5

# CNN-1D settings
CNN1D_CHANNELS = [64, 128, 256] # Feature maps per kernel
CNN1D_KERNEL   = 7              # Kernel size for each layer
CNN1D_DROPOUT  = 0.3

# CNN1D + Transformer Hybrid settings
CNNTRANS_CNN_CHANNELS = [64, 128] # CNN frontend feature maps
CNNTRANS_D_MODEL      = 128       # Transformer internal dimension
CNNTRANS_NHEAD        = 8         # Attention heads
CNNTRANS_LAYERS       = 4         # Transformer encoder layers
CNNTRANS_DIM_FF       = 256       # Feedforward size inside Transformer
CNNTRANS_DROPOUT      = 0.2

# CNN1D - ResNet settings
CNN1D_RN_CHANNELS = [64, 128, 256]
CNN1D_RN_KERNEL   = 7
CNN1D_RN_DROPOUT  = 0.3

# Temporal attention settings
TEMPORAL_PATCH_SIZE = 10
TEMPORAL_D_MODEL    = 128
TEMPORAL_NHEAD      = 4
TEMPORAL_LAYERS     = 4
TEMPORAL_DIM_FF     = 256
TEMPORAL_DROPOUT    = 0.2
