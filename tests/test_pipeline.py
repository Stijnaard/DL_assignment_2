from dl_assignment_2.data.pipeline import Pipeline
from dl_assignment_2.data.dataSegment import DataSegment, SegmentInfo

from numpy import array, ndarray, all

class Test_Pipeline:
    pl = Pipeline(3)

    m: ndarray = array([[1,2,3,4], [5,6,7,8]])

    segment: DataSegment = DataSegment(info=SegmentInfo(m, 123456, "rest", 1))

    transformed_segment = pl(segment)

    def test_correct_transformation(self):
        assert all(self.transformed_segment.data == array([[3*10**12], [7*10**12]]))


    

