
from pathlib import Path
from torch import nn


class AbsPathProvider:
    def __init__(self, results_path: Path):
        self.results_path: Path = results_path
        self.models: Path = Path.joinpath(self.results_path, "models")
        self.plots: Path = Path.joinpath(self.results_path, "plots")
        self.metrics: Path = Path.joinpath(self.results_path, "metrics")

    def get_results_path(self) -> Path:
        return self.results_path
    
    def get_models_folder_path(self) -> Path:
        return self.models
    
    def get_plots_folder_path(self) -> Path:
        return self.plots
    
    def get_metrics_folder_path(self) -> Path:
        return self.metrics
    
    def get_model_path(self, model: type[nn.Module], experiment: str) -> Path:
        """Returns the path where the model should be saved."""
        return Path.joinpath(self.models, f"{experiment}_{model.__name__}_model.pt")

    def get_plot_folder_path(self, model: type[nn.Module]) -> Path:
        """Returns the path to the folder where the plots for the given model should be saved."""
        return Path.joinpath(self.plots, model.__name__)

    def get_plot_path(self, model: type[nn.Module], experiment: str, plot_type: str) -> Path:
        """Returns the path where the plot of the given type for the given model should be saved."""
        return Path.joinpath(self.plots, model.__name__, f"{experiment}_{plot_type}.png")

    def get_metric_path(self, model: type[nn.Module], experiment: str) -> Path:
        """Returns the path where the metrics for the given model should be saved."""
        return Path.joinpath(self.metrics, f"{experiment}_{model.__name__}_metrics.pt")