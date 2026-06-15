from dl_assignment_2.data.dataSegment import DataSegment
from dl_assignment_2.data.dataset_pipeline import BaseDatasetPipeline
from dl_assignment_2.data.pipeline import BasePipeline, Pipeline
from dl_assignment_2.data.config import TASK_TYPES

from typing import Sequence, Optional

from torch.utils.data import Dataset
from torch import Tensor, tensor, accelerator, dtype, float32
from numpy import ndarray, expand_dims, concatenate

class CustomDataset(Dataset):
    feature_tensor: Tensor # (number of samples (N), timepoints (T), sensors (C))
    label_vector: Tensor
    device: str

    def __init__(self, segments: Sequence[DataSegment], dataset_pipeline: BaseDatasetPipeline, device: Optional[str] = None, element_type: dtype = float32) -> None:
        # # 1. perform the transformations if you provide a pipeline:
        # if pipeline:
        #     segments = [pipeline(segment) for segment in segments]

        # 1. perform the transformations if you provide a dataset pipeline:
        if dataset_pipeline:
            segments = dataset_pipeline(segments)
        
        # 2. set the device:
        if not device:
            self.device = accelerator.current_accelerator().type if accelerator.is_available() else "cpu" # type: ignore

        # 2. get the data from the segments:
        matrices: list[ndarray] = [expand_dims(segment.get_data(), 0) for segment in segments]
        print(f"Constructing CustomDataset with {len(matrices)} segments, each of shape {matrices[0].shape} (sensors, timepoints).")
        
        # 3. construct the full feature tensor
        self.feature_tensor: Tensor = tensor(concatenate(matrices, axis=0)).to(float32).transpose(1,2).to(device)

        # check shape of feature tensor
        print(f"Feature tensor shape: {self.feature_tensor.shape} (N, T, C).")

        # 4. construct the label vector:
        sorted_tasks = sorted(TASK_TYPES)
        self.label_vector: Tensor = tensor([sorted_tasks.index(segment.get_task()) for segment in segments]).to(float32).to(device)
        return None
    
    def __len__(self) -> int:
        return len(self.label_vector)

    def __getitem__(self, index):
        return self.feature_tensor[index,:,:], self.label_vector[index].long()
    
class ManualDataset(Dataset):
    """This Dataset class is only used for testing.
    So don't use it for anything else."""
    def __init__(self, x, y) -> None:
        if type(x) != Tensor:
            x = tensor(x).to(float32)
        if type(y) != tensor:
            y = tensor(y).to(float32)
            
        self.x = x
        self.y = y
        
    def __len__(self) -> int:
        return 1
    
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.x.to(float32), self.y.to(float32)
    
