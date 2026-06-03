from dl_assignment_2.data.dataFolderReader import FolderDataReader, ManualDataReader
from dl_assignment_2.data.subjectData import SubjectData
from dl_assignment_2.data.dataSegment import DataSegment, SegmentInfo
from dl_assignment_2.data.data_config import INTRA_TRAIN

from typing import List

from numpy import array, ndarray, all

class Test_DataFolderReader:
    x: FolderDataReader = FolderDataReader()
    
    def test_only_subject_in_get_subjectdata(self):
        subject_id: int = 105923
        subjectData: SubjectData = self.x.get_subjectData(subject_id)
        assert subjectData.get_subject_id() == subject_id
        
    def test_only_task_in_get_task_specific_data(self):
        task_type: str = "rest"
        rest_data: List[DataSegment] = self.x.get_data_for_specific_task(task_type)
        
        assert len(rest_data) == 8
        assert all(task_type == segment.get_task() for segment in rest_data)

    def test_get_correct_subject_id(self):
        subject_id: int = self.x.get_subjects()[0]
        assert subject_id == 105923

class Test_TestDataReader:
    m1: ndarray = array([[1,2], [3,4]])
    m2: ndarray = array([[10,11], [12,13]])

    s1 = DataSegment(info=SegmentInfo(m1, 1, "rest", 1))
    s1t= DataSegment(info=SegmentInfo(m1.T, 1, "rest", 2))
    s2 = DataSegment(info=SegmentInfo(m2, 2, "rest", 1))

    x: ManualDataReader= ManualDataReader([s1, s1t, s2])

    def test_reads_users_correctly(self):
        assert sorted(self.x.get_subjects()) == [1,2]

    def test_gets_segments_correct(self):
        subject_one_data: SubjectData = self.x.get_subjectData(1)
        rest_data: list[DataSegment] = subject_one_data.get_data_for_task("rest")

        assert all(rest_data[0].data == array([[1,2], [3,4]]))
        assert all(rest_data[1].data == array([[1,3], [2,4]]))
