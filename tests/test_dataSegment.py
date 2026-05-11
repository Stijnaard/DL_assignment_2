from dl_assignment_2.data.data_config import INTRA_TRAIN
from dl_assignment_2.data.dataSegment import DataSegment

from numpy import ndarray

direct_path: str = "datasets/Intra/train/rest_105923_1.h5"
indirect_path: str = f"{INTRA_TRAIN}/rest_105923_1.h5"

class Test_DataSegment:
    path: str = indirect_path
    x: DataSegment = DataSegment(path)
    
    def test_type_read(self):
        data: ndarray = self.x.get_data()
        assert type(data) == ndarray
        
    def test_correct_shape_read(self):
        data: ndarray = self.x.get_data()
        n_sensors: int = data.shape[0]
        n_recordings: int = data.shape[1]
        
        assert n_sensors == 248
        assert n_recordings == 35624
        
    def test_correct_subject_id(self):
        subject_id: int = self.x.get_subject_id()
        assert subject_id == 105923
        
    def test_correct_task(self):
        task: str = self.x.get_task()
        assert task == "rest"
        
    def test_correct_segment(self):
        segment: int = self.x.get_segment()
        assert segment == 1

    def test_indirect_path(self):
        indirect_path: str = f"{INTRA_TRAIN}/rest_105923_1.h5"
        indirect_datasegment: DataSegment = DataSegment(indirect_path)
        assert indirect_datasegment is not None
    
        
        
    