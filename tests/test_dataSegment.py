from dl_assignment_2.data.data_config import INTRA_TRAIN
from dl_assignment_2.data.dataSegment import DataSegment

from numpy import ndarray, all

indirect_path: str = f"{INTRA_TRAIN}/rest_105923_1.h5"

class Test_Method_Reading:
    path: str = indirect_path
    x: DataSegment = DataSegment(path)
    
    def test_type_read(self):
        data: ndarray = self.x.get_data()
        assert type(data) == ndarray
        
    def test_correct_shape_read(self):
        data: ndarray = self.x.get_data()
        n_sensors: int = data.shape[0]
        n_recordings: int = data.shape[1]   # shape: (248, 35624)
        
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
        
class Test_Path_Stuff:
    def test_indirect_path(self):
        indirect_path: str = f"{INTRA_TRAIN}/rest_105923_1.h5"
        indirect_datasegment: DataSegment = DataSegment(indirect_path)
        assert indirect_datasegment is not None

    def test_transform_is_for_new_obj(self):
        y: DataSegment = self.x.transform(lambda a: a*0)
        assert type(y) == DataSegment
        assert all(self.x.data) and (not all(y.data))

    def test_trim_shape_reduction(self):
        y: DataSegment = self.x.trim(one_in=2)
        assert y.data.shape[1] == 35624//2

        z: DataSegment = self.x.trim(one_in=13)
        assert z.data.shape[1] == 35624//13