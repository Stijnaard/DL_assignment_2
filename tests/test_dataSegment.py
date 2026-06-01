from dl_assignment_2.data.data_config import INTRA_TRAIN
from dl_assignment_2.data.dataSegment import DataSegment, SegmentInfo, SegmentSummary

from numpy import array, ndarray, all, zeros
import pytest

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

class Test_trim:
    indirect_path: str = f"{INTRA_TRAIN}/rest_105923_1.h5"
    x: DataSegment = DataSegment(indirect_path)
    
    def test_trim_shape_reduction(self):
        y: DataSegment = self.x.trim(n=2)
        assert y.data.shape[1] == 35624//2

        z: DataSegment = self.x.trim(n=13)
        assert z.data.shape[1] == 35624//13
        
    def test_trimming_without_rounding(self):
        test_matrix: ndarray = zeros(shape=(5, 15))
        y: DataSegment = DataSegment(info=SegmentInfo(test_matrix, 123456, "rest", 1))
        
        assert y.trim(5).shape == (5,3)
        
    def test_trimming_with_rounding(self):
        test_matrix: ndarray = zeros(shape=(5, 19))
        y: DataSegment = DataSegment(info=SegmentInfo(test_matrix, 123456, "rest", 1))
        
        assert y.trim(5).shape == (5,3)

class Test_slice:
    test_matrix: ndarray = array([[1,2],
                                  [3,4],
                                  [5,6],
                                  [7,8]])
    
    test_matrix2: ndarray = array([[1,2,3],
                                   [4,5,6],
                                   [7,8,9]])
    x1: DataSegment = DataSegment(info=SegmentInfo(test_matrix, 123456, "rest", 1))
    x2: DataSegment = DataSegment(info=SegmentInfo(test_matrix2, 123456, "rest", 1))

    def test_correctness(self):
        y: DataSegment = self.x1.slice(start=1, end=2)
        assert all(y.data == array([[3,4], [5,6]]))
        
    def test_w_negative_one_indices(self):
        y = self.x1.slice(start=2, end=-1)
        assert all(y.data == array([[5,6], [7,8]]))

    def test_w_negative_n_indices(self):
        y = self.x1.slice(start=0, end=-2)
        assert all(y.data == array([[1,2], [3,4], [5,6]]))

    def test_w_start_eq_end(self):
        y = self.x1.slice(start=1, end=1)
        assert all(y.data == array([[3,4]]))

    def test_w_columns(self):
        y = self.x2.slice(start=1,end=2, axis=1)
        assert all(y.data == array([[2,3], [5,6], [8,9]]))
  
    def test_error_start_gt_stop(self):
        with pytest.raises(ValueError):
            self.x2.slice(start=2, end=1)



        


class Test_residual:
    test_matrix: ndarray = array([[1,2,4], 
                                  [4,7,11]])
    x: DataSegment = DataSegment(info=SegmentInfo(test_matrix, 123456, "rest", 1))
    
    def test_residual_correctness(self):
        residual_x: DataSegment = self.x.get_residuals()
        assert all(residual_x.data == array([[1,2],
                                         [3,4]]))
        
class Test_summary:
    test_matrix: ndarray = array([[1,2,4], 
                                  [4,7,11]])
    x: DataSegment = DataSegment(info=SegmentInfo(test_matrix, 123456, "rest", 1))
    summary: SegmentSummary = x.summarize()
    
    def test_shape(self):
        assert self.summary.shape == (2,3)
    
    def test_min(self):
        assert self.summary.min == 1
        
    def test_max(self):
        assert self.summary.max == 11
        
    def test_mean(self):
        assert self.summary.mean == 29/6

class TestSplit:

    test_matrix: ndarray = array([[1,2,3,4],
                                  [5,6,7,8],
                                  [9,10,11,12],
                                  [13, 14, 15, 16]])
    
    x: DataSegment = DataSegment(info=SegmentInfo(test_matrix, 123466, "rest", 1))

    def test_correct_column_split(self):
        y, z = self.x.split(n = 2, axis=1)

        assert all(y.data==array([[1,2], [5,6], [9,10], [13, 14]]))
        assert all(z.data == array([[3,4], [7,8], [11, 12], [15, 16]]))

    def test_correct_row_split(self):
        y, z = self.x.split(n=2, axis=0)

        assert all(y.data == array([[1,2,3,4], [5,6,7,8]]))
        assert all(z.data == array([[9,10,11,12], [13, 14, 15, 16]]))

    def test_error_when_uneven(self):
        test_matrix = array(range(1,26)).reshape((5,5))
        x = DataSegment(info=SegmentInfo(test_matrix, 123456, "rest", 1))

        with pytest.raises(ValueError):
            y, z= x.split(n=2, axis=0)




