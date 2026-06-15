# TODO:
# - untangle plotting and evaluation code

from pathlib import Path
from typing import Any, Callable

from sklearn.metrics import accuracy_score
import torch
from torch import cuda, nn
from torch.utils.data import DataLoader

from dl_assignment_2.data.config import TASK_TYPES
from dl_assignment_2.data.absPathProvider import AbsPathProvider as DataAbsPathProvider
from dl_assignment_2.results.absPathProvider import AbsPathProvider as ResultsAbsPathProvider
from dl_assignment_2.data.dataFolderReader import FolderDataReader
from dl_assignment_2.data.pipeline import Pipeline
from dl_assignment_2.modeling.dataset import CustomDataset
from dl_assignment_2.modeling.evaluation import Evaluator
from dl_assignment_2.results.plots import plot_compare_accuracies, plot_confusion_matrix, plot_intra_vs_cross


class TestingSuite:
    """This class is responsible for evaluating a trained model on the test set."""
    experiment: str
    pipeline: Pipeline
    device: str
    results_path: Path

    _test_loaders: list[DataLoader]
    
    def __init__(self, experiment: str, results_path: Path, data_pipeline: Pipeline = Pipeline(trim_n=8), device: str | None = None) -> None:
        self.experiment = experiment
        self.device = device or ("cuda" if cuda.is_available() else "cpu")
        self.results_path = results_path
        self.pipeline = data_pipeline

        # Path providers for data and results
        self._data_path_provider = DataAbsPathProvider()
        self._results_path_provider = ResultsAbsPathProvider(results_path)
        
        # Load the test data based on the experiment type
        self._test_loaders: list[DataLoader] = []
        self._Load_test_data()

    def _Load_test_data(self) -> None:
        if self.experiment == "intra":
            test_data_paths = [self._data_path_provider.get_intra_test_path()]
        else:
            test_data_paths = [self._data_path_provider.get_cross_test_path(i + 1) for i in range(3)]

        for test_data_root in test_data_paths:
            test_reader = FolderDataReader(test_data_root)
            test_segments = []
            for task in TASK_TYPES:
                task_segments = test_reader.get_data_for_specific_task(task)
                test_segments.extend(task_segments)

            test_dataset = CustomDataset(test_segments, pipeline=self.pipeline, device=self.device)
            test_loader = DataLoader(test_dataset, batch_size=8)
            self._test_loaders.append(test_loader)

    def _plot_test(self, evaluator: Evaluator, model: nn.Module, test_index: int, show_plots: bool, save_plots: bool):
        """Plots all defined metrics and saves the plots if specified."""
        plot_folder_path = self._results_path_provider.get_plot_folder_path(type(model))
        plot_folder_path.mkdir(parents=True, exist_ok=True)

        # confusion matrix for test set:
        confusion_plot_path = self._results_path_provider.get_plot_path(type(model), self.experiment, f"test_{test_index}_confusion_matrix")
        evaluator.get_metric(model, plot_confusion_matrix, show=show_plots, save_path=confusion_plot_path if save_plots else None)

    def _test_evaluation(self, model: nn.Module, metric_fns: list[Callable], show_plots: bool, save_plots: bool) -> dict[str, float]:
        """Evaluates the model on the test set and plots the confusion matrix."""
        # Choose folder based on experiment type
        if self.experiment == "intra":
            test_data_roots = [self._data_path_provider.get_intra_test_path()]
        else:
            test_data_roots = [self._data_path_provider.get_cross_test_path(i + 1) for i in range(3)]

        # evaluate on each test set and create plots
        metrics: dict[str, Any] = {}
        for i, _ in enumerate(test_data_roots):
            test_evaluator = Evaluator(self._test_loaders[i], device=self.device)

            for metric_fn in metric_fns:
                 metric_value = test_evaluator.get_metric(model, metric_fn)
                 metrics[f"{metric_fn.__name__}_test_{i+1}"] = metric_value


            # create plots of test set evaluation
            self._plot_test(test_evaluator, model, i+1, show_plots, save_plots)
            
        return metrics
    
    def _save_metrics(self, model_type: type[nn.Module], metrics: dict[str, float]):
        """Saves the given metrics to a file."""
        metrics_path = self._results_path_provider.get_metric_path(model_type, self.experiment)
        if metrics_path.exists():
            existing_metrics = torch.load(metrics_path, map_location=self.device)
            existing_metrics.update(metrics)
            torch.save(existing_metrics, metrics_path)
        else:
            torch.save(metrics, metrics_path)


    def test_model(self, model_type: type[nn.Module], metric_fns: list[Callable], show_plots: bool=False, save_plots: bool=False):
        """Evaluates the given model on the test set."""
        #c_in, seq_len = self._test_loaders[0].dataset[0][0].shape # hacky as hell way to do this, but it works
        T, C = self._test_loaders[0].dataset[0][0].shape


        # Load the model if a path or model type is given
        model_load_path = self._results_path_provider.get_model_path(model_type, self.experiment)
        model = model_type(c_in=C, c_out=len(TASK_TYPES), seq_len=T).to(self.device)
        model.load_state_dict(torch.load(model_load_path, map_location=self.device))

        # Evaluate the model and create plots
        metrics = self._test_evaluation(model, metric_fns, show_plots, save_plots)

        # Add test metrics to existing metrics file or create a new one if it doesn't exist
        self._save_metrics(model_type, metrics)
    

    ### 
    # FROM HERE ON THE CODE WILL BE A MESS
    ###
    def get_model_accuracies(self, model_types: list[type[nn.Module]], experiment: str) -> dict[str, float]:
        """Returns a dictionary of model names and their corresponding accuracies."""
        model_accuracies = {}
        for model_type in model_types:
            # load metrics from the results folder
            metrics_path = self._results_path_provider.get_metric_path(model_type, experiment)
            metrics = torch.load(metrics_path, map_location=self.device)
            if experiment == "intra":
                metric_names = [f"{accuracy_score.__name__}_test_1"]
            else:
                metric_names = [f"{accuracy_score.__name__}_test_{i+1}" for i in range(len(self._test_loaders))]

            accuracy_accumulated = 0
            for metric_name in metric_names:
                accuracy_accumulated += metrics[metric_name]

            model_accuracies[model_type.__name__] = accuracy_accumulated / len(metric_names)

        return model_accuracies

    def compare_models(self, model_types: list[type[nn.Module]], show_plots: bool=False, save_plots: bool=False):
        """Compares multiple models on the test set."""
        model_accuracies = self.get_model_accuracies(model_types, self.experiment)

        # create comparison plot
        plot_folder_path = self._results_path_provider.get_plots_folder_path()
        plot_folder_path.mkdir(parents=True, exist_ok=True)
        comparison_plot_path = plot_folder_path / f"{self.experiment}_model_comparison.png"
        plot_compare_accuracies(model_accuracies, experiment=self.experiment, show=show_plots, save_path=comparison_plot_path if save_plots else None)
        

    def compare_models_intra_vs_cross(self, model_types: list[type[nn.Module]], show_plots: bool=False, save_plots: bool=False):
        """Compares multiple models on the test set for intra vs cross subject generalization."""
        intra_accuracies = self.get_model_accuracies(model_types, "intra")
        cross_accuracies = self.get_model_accuracies(model_types, "cross")

        # create comparison plot
        plot_folder_path = self._results_path_provider.get_plots_folder_path()
        plot_folder_path.mkdir(parents=True, exist_ok=True)
        comparison_plot_path = plot_folder_path / f"intra_vs_cross_model_comparison.png"
        plot_intra_vs_cross(intra_accuracies, cross_accuracies, show=show_plots, save_path=comparison_plot_path if save_plots else None)

