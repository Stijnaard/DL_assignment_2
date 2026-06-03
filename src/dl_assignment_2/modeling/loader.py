from dl_assignment_2.data.dataSegment import DataSegment
from dl_assignment_2.data.pipeline import Pipeline

from typing import Sequence, Optional, Callable

from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, segments: Sequence[DataSegment], pipeline: Optional[Pipeline] = None) -> None:
        if pipeline:
            for segment in segments:
                segment = pipeline(segment)
        
        

    def __len__(self) -> int:
        pass

    def __getitem__(self, index):
        pass
