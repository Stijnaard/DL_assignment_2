# TODO:
# - Implement (Niels') better training and preprocessing
# - Model saving
# - Plot saving

from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sklearn.metrics import accuracy_score
import torch
from torch import cuda, nn, optim
from torch.utils.data import DataLoader

from dl_assignment_2.data.config import TASK_TYPES
from dl_assignment_2.data.absPathProvider import AbsPathProvider
from dl_assignment_2.data.dataFolderReader import FolderDataReader
from dl_assignment_2.data.pipeline import Pipeline
from dl_assignment_2.modeling.dataset import CustomDataset
from dl_assignment_2.modeling.trainer import TrainConfig, Trainer
from dl_assignment_2.modeling.evaluation import Evaluator
from dl_assignment_2.metrics.confusionMatrix import plot_confusion_matrix

from dl_assignment_2.Casper_models.InceptionTime import InceptionTime
from dl_assignment_2.Niels_models import StackedLSTM, RNNClassifier, GRUClassifier, EEGNet, CNN1DClassifier, CNNTransformer, CNN1DResNet

class TrainingSuite:
    """This class is responsible for training the model on the training set, evaluating it on the validation set, and finally evaluating it on the test set."""
    def __init__(self):
        self.path_provider = AbsPathProvider()

    def _load_data(self, intra_or_cross: str="intra"):
        #reader = FolderDataReader(self.path_provider.get_intra_train_path())
        if intra_or_cross == "intra":
            reader = FolderDataReader(self.path_provider.get_intra_train_path())
        else:
            reader = FolderDataReader(self.path_provider.get_cross_train_path())

        rng = random.Random(0)
        train_segments = []
        valid_segments = []
        for task in TASK_TYPES:
            task_segments = reader.get_data_for_specific_task(task)
            rng.shuffle(task_segments)
            valid_segments.append(task_segments[0])
            train_segments.extend(task_segments[1:4])

        rng.shuffle(train_segments)
        rng.shuffle(valid_segments)

        return train_segments, valid_segments
    
    def _make_plots(self, trainer, show_plots, save_plots):
        if show_plots:
            trainer.plot_accuracy()
            trainer.plot_losses()
            trainer.evaluate(accuracy_score)
            trainer.evaluate(plot_confusion_matrix, show=show_plots, save_path=f"confusion_matrix_dev.png" if save_plots else None)

    def _test_evaluation(self, model: nn.Module, intra_or_cross: str, pipeline: Pipeline, device: str, show_plots: bool, save_plots: bool):
        """Evaluates the model on the test set and plots the confusion matrix."""
        # 
        if intra_or_cross == "intra":
            test_data_roots = [self.path_provider.get_intra_test_path()]
        else:
            test_data_roots = [self.path_provider.get_cross_test_path(i + 1) for i in range(3)]

        for i, test_data_root in enumerate(test_data_roots):
            #test_data_root = self.path_provider.get_intra_test_path()
            test_data_root = self.path_provider.get_cross_test_path(i + 1)
            test_reader = FolderDataReader(str(test_data_root))
            test_segments = []
            for task in TASK_TYPES:
                task_segments = test_reader.get_data_for_specific_task(task)
                test_segments.extend(task_segments)
            
            test_dataset = CustomDataset(test_segments, pipeline=pipeline, device=device)
            test_loader = DataLoader(test_dataset, batch_size=8)

            test_evaluator = Evaluator(test_loader, device=device)

            test_acc = test_evaluator.get_metric(model, accuracy_score)
            test_evaluator.get_metric(
                model, 
                plot_confusion_matrix, 
                show=show_plots, 
                save_path=f"confusion_matrix_test{i+1}.png" if save_plots else None)
            
            print(f"test accuracy: {test_acc:.4f}")

    def train_model(self, 
                    model_type: type[nn.Module], 
                    intra_or_cross: str, 
                    device: str, 
                    train_config: TrainConfig, 
                    save_model: bool=False, 
                    show_plots: bool=False, 
                    save_plots: bool=False
                    ):
        """
        Trains the model on the training set, evaluates it on the validation set, and finally evaluates it on the test set.
        """
        # Load the training and validation data
        train_segments, valid_segments = self._load_data(intra_or_cross)

        pipeline = Pipeline(trim_n=8)
        
        train_dataset = CustomDataset(train_segments, pipeline=pipeline, device=device)
        valid_dataset = CustomDataset(valid_segments, pipeline=pipeline, device=device) if valid_segments else None

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=8) if valid_dataset else None

        # Initialize the model, trainer, and evaluator
        sample_x, _ = train_dataset[0]
        model = model_type(c_in=sample_x.shape[0], c_out=len(TASK_TYPES), seq_len=sample_x.shape[1])

        trainer = Trainer(model, train_loader, train_config, eval_data=valid_loader, device=device)
        trainer.train()

        # training and validation evaluation:
        print(f"training accuracy: {trainer.train_accuracies[-1]:.4f}")
        if valid_loader:
            print(f"validation accuracy: {trainer.evaluate(accuracy_score):.4f}")

        # plot training and validation metrics:
        self._make_plots(trainer, show_plots, save_plots)

        # test set evaluation:
        self._test_evaluation(model, intra_or_cross, pipeline, device, show_plots, save_plots)

        # model saving:
        if save_model:
            model_save_path = f"{model_type.__name__}_{intra_or_cross}_model.pt"
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")
    
if __name__ == "__main__":
    device = "cuda" if cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    config = TrainConfig(
        epochs=12,
        loss_func=nn.CrossEntropyLoss,
        #optimizer=optim.Adam,
        optimizer=optim.AdamW,
        learning_rate=3e-4,
    )

    cnn = CNN1DClassifier()
    #inception_time = InceptionTime(c_in=6, c_out=5, seq_len=1500)
    
    training_suite = TrainingSuite()
    training_suite.train_model(model=cnn, device=device, train_config=config, show_plots=True)