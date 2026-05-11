from dl_assignment_2.data.dataFolderReader import DataFolderReader
from dl_assignment_2.data.subjectData import SubjectData
from dl_assignment_2.data.dataSegment import DataSegment
from dl_assignment_2.data.data_config import INTRA_TRAIN

from typing import List

class Test_DataFolderReader:
    x: DataFolderReader = DataFolderReader()
    
    def test_only_subject_in_get_subjectdata(self):
        subject_id: int = 105923
        subjectData: SubjectData = self.x.get_subjectData_for_subject(subject_id)
        assert subjectData.get_subject_id() == subject_id
        
    def test_only_task_in_get_task_specific_data(self):
        task_type: str = "rest"
        rest_data: List[DataSegment] = self.x.get_data_for_specific_task(task_type)
        
        assert len(rest_data) == 8
        assert all(task_type == segment.get_task() for segment in rest_data)