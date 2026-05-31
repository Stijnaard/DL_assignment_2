from dl_assignment_2.data.data_config import INTRA_TRAIN
from dl_assignment_2.data.dataSegment import DataSegment, DataSegmentInfo

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
    indirect_path: str = f"{INTRA_TRAIN}/rest_105923_1.h5"
    x: DataSegment = DataSegment(indirect_path)

    def test_transform_is_for_new_obj(self):
        y: DataSegment = self.x.transform(lambda a: a*0)
        assert type(y) == DataSegment
        assert all(self.x.data) and (not all(y.data))

    def test_trim_shape_reduction(self):
        y: DataSegment = self.x.trim_n_rows(n=2)
        assert y.data.shape[1] == 35624//2

        z: DataSegment = self.x.trim_n_rows(n=13)
        assert z.data.shape[1] == 35624//13

from numpy import array, all
class Test_residual:
    test_matrix: ndarray = array([[1,2,4], 
                                  [4,7,11]])
    x: DataSegment = DataSegment(dataSegmentInfo=DataSegmentInfo(test_matrix, 123456, "rest", 1))
    
    def test_residual_correctness(self):
        residual_x: DataSegment = self.x.get_residuals()
        
        assert all(residual_x.data == array([[1,2],
                                         [3,4]]))
        
        