from dl_assignment_2.data.dataSegment import DataSegment

from abc import ABC, abstractmethod

class BasePipeline(ABC):
    trim_n: int

    def __init__(self, trim_n: int = 8) -> None:
        self.trim_n: int = trim_n
        return None
    
    @abstractmethod
    def __call__(self, segment: DataSegment) -> DataSegment:
        pass

class Pipeline(BasePipeline):
    def __call__(self, segment: DataSegment) -> DataSegment:
        new_segment = segment.copy()
        new_segment = new_segment.transform()
        new_segment = new_segment.trim(self.trim_n)
        return new_segment
    
class ResidualPipeline(Pipeline):
    def __call__(self, segment: DataSegment) -> DataSegment:
        super().__call__(segment)
        segment

    