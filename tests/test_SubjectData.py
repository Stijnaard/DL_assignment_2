from dl_assignment_2.data.subjectData import SubjectData
from dl_assignment_2.data.data_config import INTRA_TRAIN



class Test_SubjectData:
    def test_only_subject_specific_data(self):
        x: SubjectData = SubjectData()