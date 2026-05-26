"""
Model registry - maps name strings to model classes.
Add a new model here to make it available in main.py.
"""
from .stacked_lstm    import StackedLSTM
from .rnn             import RNNClassifier
from .gru             import GRUClassifier
from .eegnet          import EEGNet
from .cnn1d           import CNN1DClassifier
from .cnn_transformer import CNNTransformer

# Simple lookup dict used by get_model()
MODEL_CLASSES = {
    "lstm":            StackedLSTM,
    "rnn":             RNNClassifier,
    "gru":             GRUClassifier,
    "eegnet":          EEGNet,
    "cnn1d":           CNN1DClassifier,
    "cnn_transformer": CNNTransformer
}

def get_model(name: str):
    """
    Instantiate a model by name string.

    Usage
    -----
    model = get_model("gru")          # uses default config
    model = get_model("cnn_transformer")
    """
    name = name.lower()
    if name not in MODEL_CLASSES:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available: {list(MODEL_CLASSES.keys())}"
        )
    return MODEL_CLASSES[name]()
