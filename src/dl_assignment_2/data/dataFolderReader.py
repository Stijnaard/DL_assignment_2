from dl_assignment_2.data.dataSegment import DataSegment
from dl_assignment_2.data.subjectData import SubjectData
from dl_assignment_2.data.data_config import TASK_TYPES, INTRA_TRAIN

from typing import Dict, List
from os import scandir

class DataFolderReader:
    id_to_subjectData_map: Dict[int, SubjectData] = {}
    
    def __init__(self, dataset_path: str = INTRA_TRAIN) -> None:
        self.id_to_subjectData_map: Dict[int, SubjectData] = {}
        
        for path in scandir(dataset_path):
            if not(path.name.endswith(".h5")):
                continue
            
            file_path: str = f"{dataset_path}/{path.name}"
            print("test")
            print(file_path)
            segment: DataSegment = DataSegment(file_path)
            
            subject_id: int = segment.get_subject_id()
            
            if subject_id not in self.id_to_subjectData_map:
                subjectData: SubjectData = SubjectData(subject_id)
                subjectData.add_segment(segment)
                
                self.id_to_subjectData_map[subject_id] = subjectData
            
            else:
                self.id_to_subjectData_map[subject_id].add_segment(segment)

    def get_subjectData_for_subject(self, subject_id: int) -> SubjectData:
        if subject_id in self.id_to_subjectData_map:
            return self.id_to_subjectData_map[subject_id]
        else:
            raise ValueError(f"subject {subject_id} not found in DataFolder")
        
    def get_data_for_specific_task(self, task: str) -> List[DataSegment]:
        task_specific_segments: List[DataSegment] = []
        if task not in TASK_TYPES:
            raise ValueError(f"the task queried for does not exist: {task}")
        
        for subjectData in self.id_to_subjectData_map.values():
            task_specific_segments.extend(subjectData.get_data_for_task(task))
            
        return task_specific_segments
    