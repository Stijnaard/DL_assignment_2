
from pathlib import Path
import random

from sklearn.metrics import accuracy_score
import torch
from torch import cuda, nn
from torch.utils.data import DataLoader

from dl_assignment_2.data.config import TASK_TYPES
from dl_assignment_2.data.dataFolderReader import FolderDataReader
from dl_assignment_2.data.pipeline import Pipeline
from dl_assignment_2.data.absPathProvider import AbsPathProvider as DataAbsPathProvider
from dl_assignment_2.results.absPathProvider import AbsPathProvider as ResultsAbsPathProvider
from dl_assignment_2.results.plots import plot_confusion_matrix
from dl_assignment_2.modeling.dataset import CustomDataset
from dl_assignment_2.modeling.trainer import TrainConfig, Trainer

class TrainingSuite:
    """This class is responsible for training models on the training set, evaluating it on the validation set, and finally evaluating it on the test set."""
    experiment: str
    pipeline: Pipeline
    device: str
    results_path: Path
    
    _train_loader: DataLoader
    _valid_loader: DataLoader

    _data_path_provider: DataAbsPathProvider
    _results_path_provider: ResultsAbsPathProvider

    def __init__(self, experiment: str, results_path: Path, data_pipeline: Pipeline = Pipeline(trim_n=8), device: str | None = None) -> None:
        self.experiment = experiment
        self.device = device or ("cuda" if cuda.is_available() else "cpu")
        self.results_path = results_path
        self.pipeline = data_pipeline

        # Path providers for data and results
        self._data_path_provider = DataAbsPathProvider()
        self._results_path_provider = ResultsAbsPathProvider(results_path)

        # Load the training and validation data
        self._load_train_data()

    def _load_train_data(self):
        if self.experiment == "intra":
            reader = FolderDataReader(self._data_path_provider.get_intra_train_path())
        else:
            reader = FolderDataReader(self._data_path_provider.get_cross_train_path())

        rng = random.Random(0)
        train_segments = []
        valid_segments = []
        for task in TASK_TYPES:
            task_segments = reader.get_data_for_specific_task(task)
            rng.shuffle(task_segments)
            valid_segments.append(task_segments[0])
            train_segments.extend(task_segments[1:4])

        rng.shuffle(train_segments)
        rng.shuffle(valid_segments)

        train_dataset = CustomDataset(train_segments, pipeline=self.pipeline, device=self.device)
        valid_dataset = CustomDataset(valid_segments, pipeline=self.pipeline, device=self.device) if valid_segments else None

        self.train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        self.valid_loader = DataLoader(valid_dataset, batch_size=8) if valid_dataset else None
    
    def _plot_training(self, trainer: Trainer, show_plots: bool, save_plots: bool):
        """Plots all defined metrics and saves the plots if specified."""
        plot_folder_path = self._results_path_provider.get_plot_folder_path(type(trainer.model))
        plot_folder_path.mkdir(parents=True, exist_ok=True)
        
        accuracy_plot_path = self._results_path_provider.get_plot_path(type(trainer.model), self.experiment, "val_accuracy")
        trainer.plot_accuracy(show=show_plots, save_path=accuracy_plot_path if save_plots else None)
        loss_plot_path = self._results_path_provider.get_plot_path(type(trainer.model), self.experiment, "val_loss")
        trainer.plot_losses(show=show_plots, save_path=loss_plot_path if save_plots else None)

        # confusion matrix for validation set
        confusion_plot_path = self._results_path_provider.get_plot_path(type(trainer.model), self.experiment, "val_confusion_matrix")
        trainer.evaluate(plot_confusion_matrix, show=show_plots, save_path=confusion_plot_path if save_plots else None)

    def _save_metrics(self, trainer: Trainer, model_type: type[nn.Module]):
        metrics_save_path = self._results_path_provider.get_metric_path(model_type, self.experiment)
        metrics_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "train_accuracies": trainer.train_accuracies,
            "train_losses": trainer.train_losses,
            "validation_accuracies": trainer.dev_accuracies,
        }, metrics_save_path)
        print(f"Metrics saved to {metrics_save_path}")

    def _save_model(self, model: nn.Module, model_type: type[nn.Module]):
        model_save_path = self._results_path_provider.get_model_path(model_type, self.experiment)
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

    def train_model(self, 
                    model_type: type[nn.Module], 
                    train_config: TrainConfig, 
                    save_model: bool=False, 
                    show_plots: bool=False, 
                    save_plots: bool=False,
                    save_metrics: bool=False
                    ):
        """
        Trains the model on the training set and evaluates it on the validation set.
        """
        # instantiate the model with the correct input and output dimensions
        c_in, seq_len = self.train_loader.dataset[0][0].shape 
        model = model_type(c_in=c_in, c_out=len(TASK_TYPES), seq_len=seq_len).to(self.device)

        # train the model
        trainer = Trainer(model, self.train_loader, train_config, eval_data=self.valid_loader, device=self.device)
        trainer.train()

        # training and validation evaluation
        train_acc = trainer.train_accuracies[-1]
        train_loss = trainer.train_losses[-1]
        valid_acc = trainer.evaluate(accuracy_score)

        print(f"training accuracy: {train_acc:.4f}")
        print(f"training loss: {train_loss:.4f}")
        print(f"validation accuracy: {valid_acc:.4f}")

        # metric saving
        if save_metrics:
            self._save_metrics(trainer, model_type)

        # plot training and validation metrics
        self._plot_training(trainer, show_plots, save_plots)

        # # test set evaluation and plotting
        # self._test_evaluation(model, show_plots, save_plots)

        # model saving
        if save_model:
            self._save_model(model, model_type)
