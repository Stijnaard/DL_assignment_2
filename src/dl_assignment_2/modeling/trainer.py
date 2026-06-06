from dl_assignment_2.modeling.evaluation import Evaluator

from torch import nn, Tensor, no_grad, cuda
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from dataclasses import dataclass
from typing import Callable, Optional

@dataclass
class TrainConfig:
    epochs: int
    loss_func: nn.Module
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
        self.model.train()
        for _ in range(1, self.epochs+1):
            self.train_loop()
            
            with no_grad():
                train_acc: float = self.train_evaluator.get_metric(self.model, accuracy_score)
                self.train_accuracies.append(train_acc)
        
                if self.dev_evaluator:
                    dev_acc: float = self.dev_evaluator.get_metric(self.model, accuracy_score)
                    self.dev_accuracies.append(dev_acc)
                    
        return None

    def train_loop(self) -> None:
        for X, y in self.data:
            X: Tensor = X.to(self.device)
            y: Tensor = y.to(self.device)
            
            pred: Tensor = self.model(X)

            loss =self.loss_func(pred, y)
            self.train_losses.append(loss)
            
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
    def _set_device(self, device: Optional[str]) -> None:
        if device:
            self.device: str = device
        else:
            self.device = "cuda" if cuda.is_available() else "cpu"
            
    def evaluate(self, metric: Callable) -> float:
        if not self.dev_evaluator:
            raise ValueError("no dev data was given in the init")
        
        metric_score: float = self.dev_evaluator.get_metric(self.model, metric)
        
        return metric_score
        
    

