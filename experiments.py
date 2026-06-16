
from dataclasses import dataclass, field
from typing import Optional

from torch import nn, optim

from pathlib import Path
import sys



ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sklearn.metrics import accuracy_score
from torch import cuda, nn, optim

from dl_assignment_2.Casper_models.InceptionTime import InceptionTime
from dl_assignment_2.Niels_models import StackedLSTM, RNNClassifier, GRUClassifier, EEGNet, CNN1DClassifier, CNNTransformer, CNN1DResNet
from dl_assignment_2.Stijn_models import MEG1DCNN3

from dl_assignment_2.modeling.trainer import TrainConfig
from dl_assignment_2.trainingSuite import SessionConfig, TrainingSuite
from dl_assignment_2.testingSuite import TestingSuite
from dl_assignment_2.data.pipeline import Pipeline
from dl_assignment_2.data.dataset_pipeline import WindowingPipeline, DatasetPipeline


# Base parameters
MODELS: list[type[nn.Module]] = [MEG1DCNN3, CNN1DClassifier, CNN1DResNet, InceptionTime] #[StackedLSTM, RNNClassifier, GRUClassifier, EEGNet, CNN1DClassifier, CNNTransformer, CNN1DResNet, InceptionTime] #[StackedLSTM] #[CNN1DResNet] #, RNNClassifier]

@dataclass
class ExperimentConfig:
    experiment_name: str
    models: list[type[nn.Module]] = field(default_factory=lambda: MODELS)
    # Preprocessing parameters
    use_windowing: bool = True
    window_size: int = 512
    window_stride: int = 512
    downsample_factor: int = 4
    # Training parameters
    epochs: int = 40
    loss_func: type[nn.Module] = nn.CrossEntropyLoss
    optimizer: type[optim.Optimizer] = optim.AdamW
    label_smoothing: Optional[float] = None
    learning_rate: float = 0.0003
    weight_decay: float = 5e-4
    dropout_rate: float = 0.3


def run_experiment(config: ExperimentConfig):
    # Set up the experiment based on the provided configuration
    print(f"Running experiment: {config.experiment_name}")
    print(f"Use windowing: {config.use_windowing}")
    print(f"Window size: {config.window_size}")
    print(f"Window stride: {config.window_stride}")
    print(f"Downsample factor: {config.downsample_factor}")
    print(f"Epochs: {config.epochs}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Weight decay: {config.weight_decay}")
    print(f"Dropout rate: {config.dropout_rate}")

    device = "cuda" if cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if config.use_windowing:
        preprocessing_pipeline = WindowingPipeline(trim_n=config.downsample_factor, window_size=config.window_size, window_stride=config.window_stride)
    else:
        preprocessing_pipeline = DatasetPipeline(trim_n=config.downsample_factor)

    session_config = SessionConfig(
        experiment="intra", # Do intra first, then cross
        pipeline=preprocessing_pipeline,
        results_path=ROOT / "results" / "experiments" / config.experiment_name,
        device=device
    )

    train_config = TrainConfig(
        epochs=config.epochs,
        loss_func=config.loss_func,
        optimizer=config.optimizer,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        label_smoothing=config.label_smoothing,
    )

    # Training and testing on intra-subject data
    training_suite = TrainingSuite(session_config=session_config)
    print(f"Training suite created with {training_suite._train_loader.dataset[0][0].shape} input shape and {len(training_suite._train_loader.dataset)} training samples.")
    print(f"Validation suite created with {training_suite._valid_loader.dataset[0][0].shape} input shape and {len(training_suite._valid_loader.dataset)} validation samples.")
    testing_suite = TestingSuite(session_config=session_config)
    for model_cls in MODELS:
        print(f"Training {model_cls.__name__} on intra-subject data...")
        training_suite.train_model(model_type=model_cls, 
                                   train_config=train_config, 
                                   dropout=config.dropout_rate,
                                   show_plots=False, 
                                   save_plots=True,
                                   save_model=True,
                                   save_metrics=True)
        
        testing_suite.test_model(model_type=model_cls,
                                 metric_fns=[accuracy_score],
                                 show_plots=False,
                                 save_plots=True)
    
    testing_suite.compare_models(model_types=MODELS, show_plots=False, save_plots=True)

    session2_config = session_config
    session2_config.experiment = "cross"

    # Training and testing on cross-subject data
    training_suite = TrainingSuite(session_config=session2_config)
    testing_suite = TestingSuite(session_config=session2_config)
    for model_cls in MODELS:
        print(f"Training {model_cls.__name__} on cross-subject data...")
        training_suite.train_model(model_type=model_cls, 
                                   train_config=train_config, 
                                   dropout=config.dropout_rate,
                                   show_plots=False, 
                                   save_plots=True,
                                   save_model=True,
                                   save_metrics=True)
        
        testing_suite.test_model(model_type=model_cls,
                                 metric_fns=[accuracy_score],
                                 show_plots=False,
                                 save_plots=True)
    
    testing_suite.compare_models(model_types=MODELS, show_plots=False, save_plots=True)
    
    testing_suite.compare_models_intra_vs_cross(model_types=MODELS, show_plots=False, save_plots=True)



window_size_experiment = [
    ExperimentConfig(
        experiment_name="window_size_128",
        window_size=128,
        window_stride=128,
    ),
    # ExperimentConfig(
    #     experiment_name="window_size_256",
    #     window_size=256,
    #     window_stride=256,
    # ),
    ExperimentConfig(
        experiment_name="window_size_512",
        window_size=512,
        window_stride=512,
    ),
    # ExperimentConfig(
    #     experiment_name="window_size_1024",
    #     window_size=1024,
    #     window_stride=1024,
    # ),
    ExperimentConfig(
        experiment_name="window_size_2048",
        window_size=2048,
        window_stride=2048,
    ),
    ExperimentConfig(
        experiment_name="no_windows",
        use_windowing=False,
    ),
]

dropout_experiment = [
    ExperimentConfig(
        experiment_name="dropout_0.0",
        dropout_rate=0.0,
    ),
    ExperimentConfig(
        experiment_name="dropout_0.1",
        dropout_rate=0.1,
    ),
    ExperimentConfig(
        experiment_name="dropout_0.3",
        dropout_rate=0.3,
    ),
    ExperimentConfig(
        experiment_name="dropout_0.5",
        dropout_rate=0.5,
    ),
    ExperimentConfig(
        experiment_name="dropout_0.7",
        dropout_rate=0.7,
    ),
]

if __name__ == "__main__":
    for experiment_config in window_size_experiment:
        run_experiment(experiment_config)

    for experiment_config in dropout_experiment:
        run_experiment(experiment_config)
