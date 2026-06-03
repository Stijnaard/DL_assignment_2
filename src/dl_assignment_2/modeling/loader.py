from dl_assignment_2.data.dataSegment import DataSegment
from dl_assignment_2.data.pipeline import Pipeline

from typing import Sequence, Optional, Callable

from torch.utils.data import Dataset
from torch import Tensor
from numpy import ndarray

class CustomDataset(Dataset):
    feature_matrix: Tensor
    label_vector: Tensor

    def __init__(self, segments: Sequence[DataSegment], pipeline: Optional[Pipeline] = None) -> None:
        # 1. perform the transformations if you provide a pipeline:
        if pipeline:
            for segment in segments:
                segment = pipeline(segment)

        # 2. get the data from the segments:
        matrices: list[ndarray] = [segment.get_data() for segment in segments]
    
        

        


    def __len__(self) -> int:
        pass

    def __getitem__(self, index):
        pass
