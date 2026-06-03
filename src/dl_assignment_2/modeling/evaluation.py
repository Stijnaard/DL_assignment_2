from typing import Sequence, Callable, Optional

from torch.utils.data import DataLoader
from torch import nn, accelerator, no_grad, Tensor

class Evaluator:
    """The Evaluator object takes care of evaluating a model with respect
    to its testing data and the given metrics. It returnss """
    def __init__(self, model: nn.Module, data: DataLoader, device: Optional[str] = None) -> None:
        if not device:
            self.device = accelerator.current_accelerator().type if accelerator.is_available() else "cpu" # type: ignore
        
        self.model: nn.Module = model.to(device)
        self.data: DataLoader = data
        return None

    def evaluate(self) -> Evaluation: # type: ignore
        self.model.eval()
        
        with no_grad():
            for X, y in self.data:
                prediction_logits: Tensor = self.model(X)
                
            

class Evaluation:
    def __init__(self, results: dict[str, float]) -> None:
        pass