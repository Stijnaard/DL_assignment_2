# Deep learning MEG Decoding
Classify brain states (Rest / Motor / Math / Memory) from MEG signals.

## Structure
```
DL_ASSIGNMENT_2/
main.py
requirements.txt
datasets/ (not on Github)
|- Intra/ - Intra has 68  windows with this setup per type.
    |- train/
    |- test/
|- Cross/   - Cross has 408 windows with this setup per type.
    |- train/
    |- test1/
    |- test2/
    |- test3/
outputs/
|- figures/
|- checkpoints/
|- results/
src/
|- config/
    |- config.py
|- data/
    │   └── loader.py        ← data loading, preprocessing, windowing
|- models/
    |- lstm.py             Bi-dir. stacked LSTM
    |- rnn.py              Bidirectional RNN
    |- gru.py              GRU with Attention Pooling
    |- eegnet.py           EEGNet (designed for neural signals)
    |- cnn1d.py            1D CNN
    |- cnn1d_resnet        1D CNN + ResNet
    |- cnn_transformer.py  CNN + Transformer Hybrid
|- training/
    |- trainer.py       ← training loop, early stopping, checkpointing
|- evaluation/
    |- plots.py         ← all figures and metrics for the paper
```

### 1. Setup
```bash
pip install -r requirements.txt and data in 'datasets/'.
```

### 2. Running
```bash
# Runs model specified in config.py
python main.py

# Train a specific model, example:
python main.py --model lstm

# Train ALL models and get a full comparison:
python main.py --model all

# Run only the Intra-subject experiment:
python main.py --model lstm --experiment intra

# Skip training, reload saved models, just regenerate plots:
python main.py --eval-only
```