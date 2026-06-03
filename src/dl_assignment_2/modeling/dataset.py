from dl_assignment_2.data.dataSegment import DataSegment
from dl_assignment_2.data.pipeline import Pipeline
from dl_assignment_2.data.config import TASK_TYPES

from typing import Sequence, Optional, Callable

from torch.utils.data import Dataset
from torch import Tensor, tensor
from numpy import ndarray, expand_dims, concatenate

class CustomDataset(Dataset):
    feature_tensor: Tensor # (N, sensors, T)
    label_vector: Tensor
    device: str

    def __init__(self, segments: Sequence[DataSegment], pipeline: Optional[Pipeline] = None) -> None:
        # 1. perform the transformations if you provide a pipeline:
        if pipeline:
            for segment in segments:
                segment = pipeline(segment)

        # 2. get the data from the segments:
        matrices: list[ndarray] = [expand_dims(segment.get_data(), 0) for segment in segments]
        
        # 3. construct the full feature tensor
        self.feature_tensor: Tensor = tensor(concatenate(matrices, axis=0))

        # 4. construct the label vector:
        sorted_tasks = sorted(TASK_TYPES)
        self.label_vector: Tensor = tensor([sorted_tasks.index(segment.get_task()) for segment in segments])
        return None
    
    def __len__(self) -> int:
        return len(self.label_vector)

    def __getitem__(self, index):
        return self.feature_tensor[index,:,:], self.label_vector[index]
    
