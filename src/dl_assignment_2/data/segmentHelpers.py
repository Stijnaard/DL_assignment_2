from dataclasses import dataclass
from numpy import ndarray

@dataclass
class SegmentInfo:
    """
    This class helps out with transferring info from one datasegment to another
    when trying to create a copy or new object based on the data of another one.

    It should not be created manually ever.
    """
    data: ndarray
    subject_id: int
    task: str
    segment: int
    
@dataclass
class SegmentSummary:
    min: float
    max: float
    shape: tuple[int, int]
    mean: float
    std: float
    segment: int
    subject_id: int
    task: str
    
def __str__(self) -> str:
    return f"{self.subject_id}'s  {self.task} Segment {self.shape} w. min: {self.min}, max: {self.max} and mean: {self.mean}"
    
