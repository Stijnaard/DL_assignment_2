
from abc import ABC, abstractmethod
import random
from typing import Sequence

from dl_assignment_2.data.dataSegment import DataSegment

class BaseDatasetPipeline(ABC):
    @abstractmethod
    def __call__(self, segments: Sequence[DataSegment]) -> list[DataSegment]:
        pass

class DatasetPipeline(BaseDatasetPipeline):
    def __init__(self, trim_n: int) -> None:
        self.trim_n = trim_n

    def __call__(self, segments: Sequence[DataSegment]) -> list[DataSegment]:
        new_segments: list[DataSegment] = []

        for segment in segments:
            # z-score normalization
            new_segment = segment.z_score_normalize()

            # trim
            new_segment = new_segment.trim(self.trim_n)

            new_segments.append(new_segment)
    
        return new_segments
    
class WindowingPipeline(DatasetPipeline):
    def __init__(self, trim_n: int, window_size: int, window_stride: int) -> None:
        super().__init__(trim_n)
        self.window_size = window_size
        self.window_stride = window_stride
        return None

    def __call__(self, segments: Sequence[DataSegment]) -> list[DataSegment]:
        segments = super().__call__(segments)

        new_segments: list[DataSegment] = []
        for segment in segments:
            new_segments.extend(segment.window(self.window_size, self.window_stride))

        return new_segments