from dl_assignment_2.data.dataSegment import DataSegment

from typing import Sequence, Optional, Callable

from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, segments: Sequence[DataSegment], transformations: Optional[Sequence[Callable]] = None) -> None:
        pass

    def __len__(self) -> int:
        pass

    def __getitem__(self, index):
        pass
