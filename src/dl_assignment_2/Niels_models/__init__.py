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
from .cnn1d_resnet    import CNN1DResNet

# Simple lookup dict used by get_model()
MODEL_CLASSES = {
    "lstm":               StackedLSTM,
    "rnn":                RNNClassifier,
    "gru":                GRUClassifier,
    "eegnet":             EEGNet,
    "cnn1d":              CNN1DClassifier,
    "cnn_transformer":    CNNTransformer,
    "cnn1d_resnet":       CNN1DResNet}

def get_model(name: str):
    """Instantiate a model by name string"""
    name = name.lower()
    if name not in MODEL_CLASSES:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available: {list(MODEL_CLASSES.keys())}"
        )
    return MODEL_CLASSES[name]()
