from typing import Sequence, Callable, Optional

from torch.utils.data import DataLoader
from torch import nn, accelerator, no_grad, Tensor, concatenate
from sklearn.metrics import accuracy_score

class Evaluator:
    """The Evaluator object takes care of evaluating a model with respect
    to its testing data and the given metrics. It returnss """
    def __init__(self, data: DataLoader, device: Optional[str] = None) -> None:
        if not device:
            self.device = accelerator.current_accelerator().type if accelerator.is_available() else "cpu" # type: ignore
        else:
            self.device = device
            
        print(f"evaluator device: {self.device}")

        self.data: DataLoader = data
        return None

    def compute_predictions(self, model: nn.Module) -> tuple[Tensor, Tensor]:
        model.eval()
        all_predictions: list[Tensor] = []
        all_labels: list[Tensor] = []

        with no_grad():
            for X, y in self.data:
                X: Tensor = X.to(self.device)
                y: Tensor = y.to(self.device)
                prediction_logits: Tensor = model(X)
                all_predictions.append(prediction_logits)
                all_labels.append(y)

        predictions: Tensor = concatenate(all_predictions, 0)
        labels: Tensor = concatenate(all_labels, 0)
        return predictions, labels
    
    def get_metric(self, model: nn.Module, metric: Callable) -> float:
        preds, labels = self.compute_predictions(model)
        preds, labels = preds.cpu(), labels.cpu()
        predicted_indices: Tensor = preds.argmax(1)
        
        return metric(labels, predicted_indices)
    
    def get_loss(self, model, loss_func: nn.Module) -> float:
        model.eval()
        all_predictions: list[Tensor] = []
        all_labels: list[Tensor] = []
        total_loss: float = 0
        
        with no_grad():
            for X, y in self.data:
                X: Tensor = X.to(self.device)
                y: Tensor = y.to(self.device)
                
                prediction_logits: Tensor = model(X)
                loss = loss_func(prediction_logits, y)
                
                total_loss += loss.item()
                
        return total_loss
