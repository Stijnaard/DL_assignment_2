from dl_assignment_2.modeling.loader import CustomDataset
from dl_assignment_2.data.dataSegment import DataSegment, SegmentInfo

from numpy import ndarray, array, all
from torch.utils.data import DataLoader

class Test_Dataset:
    m1: ndarray = array([[1,2,3,4], [5,6,7,8]])
    m2: ndarray = m1*-1

    s1: DataSegment = DataSegment(info=SegmentInfo(m1, 123456, "rest", 1))
    s2: DataSegment = DataSegment(info=SegmentInfo(m1, 123456, "task_motor", 1))

    def test_correct_feature_matrix(self):
        ds: CustomDataset = CustomDataset((self.s1, self.s2))
        x, y = ds.__getitem__(0)
        assert all(x == array([[1,2,3,4], [5,6,7,8]]))
        

    def test_correct_label(self):
        ds: CustomDataset = CustomDataset((self.s1, self.s2))
        x, y = ds.__getitem__(0)
        assert y == 3