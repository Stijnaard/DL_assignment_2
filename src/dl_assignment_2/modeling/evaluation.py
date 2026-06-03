from dl_assignment_2.modeling.model import BaseModel

from typing import Sequence, Callable

from torch.utils.data import DataLoader

class Evaluator:
    """The Evaluator object takes care of evaluating a model with respect
    to its testing data and the given metrics. It returnss """
    def __init__(self, model: BaseModel, data: DataLoader, metrics: Sequence[Callable]) -> None:
        pass

    def evaluate(self) -> Evaluation:
        pass

class Evaluation:
    pass