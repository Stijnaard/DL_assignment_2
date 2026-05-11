from dl_assignment_2.data.dataSegment import DataSegment
from typing import List


class SubjectData:
    data_segments: List[DataSegment] = []
    def __init__(self, subject_id: int, data_file_paths: List[str] = []) -> None:
        self.subject_id: int = subject_id
        self.data_segments = [DataSegment(path) for path in data_file_paths]

        self._check_data_ownership()
        return None
    
    def get_subject_id(self) -> int:
        return self.subject_id
    
    def _check_data_ownership(self) -> None:
        for dataSegment in self.data_segments:
            if dataSegment.get_subject_id() != self.subject_id:
                raise ValueError(f"data segment for: {dataSegment.get_subject_id()} does not belong to {self.subject_id} SubjectData")
        
        return None
    
    
    def get_data_for_task(self, task: str) -> List[DataSegment]:
        task_specific_segments: List[DataSegment] = []
        
        for segment in self.data_segments:
            if segment.get_task() == task:
                task_specific_segments.append(segment)
                
        return task_specific_segments
    
    
    def add_segment(self, segment: DataSegment) -> None:
        self._check_segment_ownership(segment)
        
        self.data_segments.append(segment)
        
    def _check_segment_ownership(self, dataSegment: DataSegment) -> None:
        if self.subject_id != dataSegment.get_subject_id():
            ValueError(f"data segment for: {dataSegment.get_subject_id()} does not belong to {self.subject_id} SubjectData")
        
    
        
    
        