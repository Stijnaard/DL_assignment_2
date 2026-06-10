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
from dl_assignment_2.Niels_models.cnn1d import CNN1DClassifier

class TrainingSuite:
    """This class is responsible for training the model on the training set, evaluating it on the validation set, and finally evaluating it on the test set."""
    def __init__(self):
        self.path_provider = AbsPathProvider()

    def load_data(self):
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

    def train_model(self, model, device, train_config, save_model=False, show_plots=False, save_plots=False):
        """
        Trains the model on the training set, evaluates it on the validation set, and finally evaluates it on the test set.
        """
        train_segments, valid_segments = self.load_data()

        pipeline = Pipeline(trim_n=8)
        
        train_dataset = CustomDataset(train_segments, pipeline=pipeline, device=device)
        valid_dataset = CustomDataset(valid_segments, pipeline=pipeline, device=device) if valid_segments else None

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=8) if valid_dataset else None

        #sample_x, _ = train_dataset[0]
        #model = InceptionTime(c_in=sample_x.shape[0], c_out=len(TASK_TYPES), seq_len=sample_x.shape[1])

        trainer = Trainer(model, train_loader, train_config, eval_data=valid_loader, device=device)
        trainer.train()

        print(f"training accuracy: {trainer.train_accuracies[-1]:.4f}")

        if show_plots:
            trainer.plot_accuracy()
            trainer.plot_losses()
            trainer.evaluate(accuracy_score)
            trainer.evaluate(plot_confusion_matrix, show=show_plots, save_path=f"confusion_matrix_dev.png" if save_plots else None)

        if valid_loader:
            print(f"validation accuracy: {trainer.evaluate(accuracy_score):.4f}")

        # test set evaluation:
        for i in range(3):
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

if __name__ == "__main__":
    device = "cuda" if cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    config = TrainConfig(
        epochs=12,
        loss_func=nn.CrossEntropyLoss,
        optimizer=optim.Adam,
        learning_rate=3e-4,
    )

    cnn = CNN1DClassifier()
    #inception_time = InceptionTime(c_in=6, c_out=5, seq_len=1500)
    
    training_suite = TrainingSuite()
    training_suite.train_model(model=cnn, device=device, train_config=config, show_plots=True)