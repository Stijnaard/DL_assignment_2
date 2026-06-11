from dl_assignment_2.modeling.evaluation import Evaluator

from torch import nn, Tensor, no_grad, cuda
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Callable, Optional
import time

@dataclass
class TrainConfig:
    epochs: int
    loss_func: type[nn.Module]
    optimizer: type
    learning_rate: float

class Trainer:
    def __init__(self, model: nn.Module, data: DataLoader, config: TrainConfig, eval_data: Optional[DataLoader] = None, device: Optional[str] = None) -> None:
        self._set_device(device)
        
        self.model  : nn.Module  = model.to(self.device)
        self.data   : DataLoader = data
        
        self.epochs     : int       = config.epochs
        self.loss_func  : nn.Module = config.loss_func()
        self.optimizer  : Optimizer = config.optimizer(self.model.parameters(), config.learning_rate)
        
        self.train_evaluator: Evaluator             = Evaluator(data)
        self.dev_evaluator  : Optional[Evaluator]   = Evaluator(eval_data) if eval_data else None
        
        self.train_accuracies   : list[float] = []
        self.train_losses       : list[float] = []
        self.dev_accuracies     : list[float] = []
        return None

    def train(self) -> None:
        with no_grad():
            start_train_acc: float = self.train_evaluator.get_metric(self.model, accuracy_score)
            start_train_loss: float = self.train_evaluator.get_loss(self.model, self.loss_func)
            
            self.train_accuracies.append(start_train_acc) 
            self.train_losses.append(start_train_loss)
        
        
        for epoch in range(1, self.epochs+1):
            print(f"epoch: {epoch}")
            start_time = time.time()
            
            self.train_loop()
            
            training_time = time.time()
            print(f"training time: {training_time-start_time}")
            
            with no_grad():
                start_train_acc: float = self.train_evaluator.get_metric(self.model, accuracy_score)
                self.train_accuracies.append(start_train_acc)
        
                if self.dev_evaluator:
                    dev_acc: float = self.dev_evaluator.get_metric(self.model, accuracy_score)
                    self.dev_accuracies.append(dev_acc)
                    
            eval_time = time.time()
            print(f"eval time: {eval_time - training_time}")

                    
        return None

    def train_loop(self) -> None:
        self.model.train()
        total_loss: float = 0
        
        for X, y in self.data:
            X: Tensor = X.to(self.device)
            y: Tensor = y.to(self.device)
            
            pred: Tensor = self.model(X)

            loss: Tensor = self.loss_func(pred, y)
            total_loss += loss.item()
            
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
        self.train_losses.append(total_loss)
        
    def _set_device(self, device: Optional[str]) -> None:
        if device:
            self.device: str = device
        else:
            self.device = "cuda" if cuda.is_available() else "cpu"
            
    def evaluate(self, metric: Callable, *args, **kwargs) -> float:
        if not self.dev_evaluator:
            raise ValueError("no dev data was given in the init")
        
        metric_score: float = self.dev_evaluator.get_metric(self.model, metric, *args, **kwargs)
        
        return metric_score

    def plot_accuracy(self, axis: Optional[Axes] = None, show: bool = False, save_path: Optional[str] = None) -> Axes:
        axis_given: bool = axis is not None
        
        if not axis_given:
            fig, axis = plt.subplots()
        
        axis.plot(self.train_accuracies, label="train accuracies")
        if self.dev_evaluator:
            axis.plot(self.dev_accuracies, label="dev accuracies")
        axis.legend()
        
        if not axis_given:
            if save_path:
                fig.savefig(save_path) # type: ignore
            if show:
                plt.show()
            plt.close(fig) # type: ignore
    
        return axis
    
    def plot_losses(self, axis: Optional[Axes] = None, show: bool = False, save_path: Optional[str] = None) -> Axes:
        axis_given: bool = axis is not None
        
        if not axis_given:
            fig, axis = plt.subplots()
        
        axis.plot(self.train_losses, label="train losses")  # type: ingnore
        axis.legend()
        
        if not axis_given:
            if save_path:
                plt.savefig(save_path) # type: ignore
            if show:
                plt.show()
            plt.close(fig) # type: ignore
        return axis

    