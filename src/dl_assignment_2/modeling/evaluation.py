from typing import Sequence, Callable, Optional

from torch.utils.data import DataLoader
from torch import nn, accelerator, no_grad, Tensor, concatenate

class Evaluator:
    """The Evaluator object takes care of evaluating a model with respect
    to its testing data and the given metrics. It returnss """
    def __init__(self, model: nn.Module, data: DataLoader, device: Optional[str] = None) -> None:
        if not device:
            self.device = accelerator.current_accelerator().type if accelerator.is_available() else "cpu" # type: ignore

        self.model: nn.Module = model.to(device)
        self.data: DataLoader = data

        self.predictions: Tensor = self.compute_predictions()
        return None

    def compute_predictions(self) -> Tensor:
        self.model.eval()
        all_predictions: list[Tensor] = []

        with no_grad():
            for X, _ in self.data:
                prediction_logits: Tensor = self.model(X)
                all_predictions.append(prediction_logits)

        predictions: Tensor = concatenate(all_predictions, 0)

        return predictions

class Evaluation:
    def __init__(self, results: dict[str, float]) -> None:
        pass
