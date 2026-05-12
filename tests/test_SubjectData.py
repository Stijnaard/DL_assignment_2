from dl_assignment_2.data.subjectData import SubjectData
from dl_assignment_2.data.data_config import INTRA_TRAIN
from dl_assignment_2.data.dataFolderReader import DataFolderReader


class Test_SubjectData:
    data: DataFolderReader = DataFolderReader()
    subject_id: int = data.get_subjects()[0]
    subject: SubjectData = data.get_subjectData(subject_id)
    