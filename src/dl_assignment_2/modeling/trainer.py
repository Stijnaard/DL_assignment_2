from torch import nn
from torch.utils.data import DataLoader

from dataclasses import dataclass
from typing import Callable


class Trainer:
    evaluator: 
    def __init__(self, model: nn.Module, data: DataLoader, config: TrainConfig) -> None:
        pass
    
    def train(self) -> None:
        pass

    def _train_for_epoch(self) -> None:
        pass
    
@dataclass
class TrainConfig:
    epochs: int
    loss_func: Callable
    learning_rate: float
    regularization: Callable

