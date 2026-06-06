from torch import nn
from torch.utils.data import DataLoader

from dataclasses import dataclass
from typing import Callable


class Trainer:
    def __init__(self, model: nn.Module, data: DataLoader, epochs: int, optimizer) -> None:
        self.model: nn.Module = model
        self.data: DataLoader = data

    def train(self) -> None:
        pass

    def train_loop(self) -> None:
        for X, y in self.data:
            pred: Tensor = self.model(X)

            loss: float =
        pass
@dataclass
class TrainConfig:
    epochs: int
    loss_func: Callable
    learning_rate: float
    regularization: Callable

