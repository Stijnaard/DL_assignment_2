"""tests the WithinTaskAnalysis class"""
from dl_assignment_2.data.dataSegment import DataSegment, SegmentInfo
from dl_assignment_2.analysis.task_level.WithinTaskAnalysis import WithinTaskAnalysis
from numpy import ndarray, array, all, concatenate

class Test_Instantiation:
    
    def test_full_segment(self):
        a: ndarray = array([[1],[1],[1]])
        b: ndarray = array([[2],[2],[2]])
        c: ndarray = array([[3],[3],[3]])

        a_s = DataSegment(info=SegmentInfo(a, 123456, "rest", 1))
        b_s = DataSegment(info=SegmentInfo(b, 123456, "rest", 2))
        c_s = DataSegment(info=SegmentInfo(c, 123456, "rest", 3))

        wta = WithinTaskAnalysis(segments=[a_s, b_s, c_s])

        assert all(wta.full_segment.data == array([[1,2,3], [1,2,3], [1,2,3]])) 

        
