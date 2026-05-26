# MEG Brain Decoding — Deep Learning Project (INFOMDLR)

Classify brain states (Rest / Motor / Math / Memory) from MEG signals using 5 deep learning models.

---

## Project Structure

```
meg_project/
│
├── main.py                  ← RUN THIS FILE
├── requirements.txt         ← install dependencies
│
├── datasets/                ← put your downloaded data here
│   ├── Intra/
│   │   ├── train/           (rest_105923_1.h5, task_motor_105923_1.h5, ...)
│   │   └── test/
│   └── Cross/
│       ├── train/
│       ├── test1/
│       ├── test2/
│       └── test3/
│
├── outputs/                 ← all results saved here automatically
│   ├── figures/             (PNG + PDF plots for your paper)
│   ├── checkpoints/         (saved model weights)
│   └── results/             (results_summary.csv)
│
└── src/
    ├── config/
    │   └── config.py        ← ALL settings live here (change things here!)
    ├── data/
    │   └── loader.py        ← data loading, preprocessing, windowing
    ├── models/
    │   ├── rnn.py           ← Vanilla Bidirectional RNN
    │   ├── gru.py           ← GRU with Attention Pooling
    │   ├── eegnet.py        ← EEGNet (designed for neural signals)
    │   ├── cnn1d.py         ← 1D CNN
    │   └── cnn_transformer.py ← CNN + Transformer Hybrid (recommended)
    ├── training/
    │   └── trainer.py       ← training loop, early stopping, checkpointing
    └── evaluation/
        └── plots.py         ← all figures and metrics for the paper
```

---

## Quick Start

### 1. Install dependencies
```bash
# First install PyTorch with CUDA from https://pytorch.org/get-started/locally/
# Then install the rest:
pip install -r requirements.txt
```

### 2. Place your data
Put the downloaded dataset in the `datasets/` folder so the structure matches above.

### 3. Run!
```bash
# Train the recommended model (CNN + Transformer) on both experiments:
python main.py

# Train a specific model:
python main.py --model gru
python main.py --model rnn
python main.py --model eegnet
python main.py --model cnn1d
python main.py --model cnn_transformer

# Train ALL models and get a full comparison:
python main.py --model all

# Run only the Intra-subject experiment:
python main.py --model gru --experiment intra

# Skip training, reload saved models, just regenerate plots:
python main.py --eval-only

# Override number of epochs from command line:
python main.py --model gru --epochs 30
```

---

## Models

| Model | Description | Best for |
|-------|-------------|----------|
| `rnn` | Vanilla bidirectional RNN | Baseline comparison |
| `gru` | GRU + attention pooling | Strong sequential baseline |
| `eegnet` | CNN designed for neural signals | Compact, low overfitting |
| `cnn1d` | 1D CNN with global pooling | Fast training |
| `cnn_transformer` | CNN frontend + Transformer encoder | **Best accuracy** |

---

## What gets saved

After training, `outputs/` will contain:

**Figures** (`outputs/figures/`):
- `{model}_{experiment}_curves.png` — loss and accuracy over epochs
- `{model}_{experiment}_confusion.png` — confusion matrix heatmap
- `comparison_{experiment}_bar.png` — accuracy comparison across all models
- `intra_vs_cross.png` — generalisation gap visualisation
- `class_dist_*.png` — class balance check

**Results** (`outputs/results/`):
- `results_summary.csv` — table of all accuracy and F1 scores

**Checkpoints** (`outputs/checkpoints/`):
- `{model}_{experiment}_best.pt` — saved model weights

---

## Changing Settings

Open `src/config/config.py` and edit any value. Key settings:

```python
MODEL = "cnn_transformer"   # default model
EPOCHS = 50                 # training epochs
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WINDOW_SIZE = 200           # time steps per sample (1 second at 200 Hz)
DOWNSAMPLE_FACTOR = 10      # 2034 Hz → ~200 Hz
FILES_PER_CHUNK = 8         # Cross training: files loaded at once
```

---

## Assignment Hints Implementation

| Hint | Where implemented |
|------|------------------|
| Z-score normalisation | `src/data/loader.py` → `zscore()` |
| Downsampling (2034 → 200 Hz) | `src/data/loader.py` → `downsample()` |
| Memory management (chunked loading) | `src/training/trainer.py` → `train_chunked()` |
