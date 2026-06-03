from dl_assignment_2.modeling.loader import CustomDataset
from dl_assignment_2.data.dataSegment import DataSegment, SegmentInfo

from numpy import ndarray, array
from torch.utils.data import DataLoader
from torch import tensor, all

class Test_Dataset_single_instances:
    m1: ndarray = array([[1,2,3,4], [5,6,7,8]])
    m2: ndarray = m1*-1

    s1: DataSegment = DataSegment(info=SegmentInfo(m1, 123456, "rest", 1))
    s2: DataSegment = DataSegment(info=SegmentInfo(m2, 123456, "task_motor", 1))

    ds: CustomDataset = CustomDataset((s1, s2))



    def test_correct_feature_matrix(self):
        x, _ = self.ds.__getitem__(0)
        assert all(x == tensor([[1,2,3,4], [5,6,7,8]]))
        

    def test_correct_label(self):
        _, y = self.ds.__getitem__(0)
        assert y == 0   # 0 = rest (moves alphabetically)

    def test_correct_length(self):
        assert len(self.ds) == 2

class Test_Dataset_Batches:
    m1: ndarray = array([[1,2,3,4], [5,6,7,8]])
    m2: ndarray = m1*-1

    s1: DataSegment = DataSegment(info=SegmentInfo(m1, 123456, "rest", 1))
    s2: DataSegment = DataSegment(info=SegmentInfo(m2, 123456, "task_motor", 1))

    ds: CustomDataset = CustomDataset((s1, s2))
    dl: DataLoader = DataLoader(ds, 2)

    def test_correct_feature_batch(self):
        for X, _ in self.dl:
            assert all(X == tensor([[[1,2,3,4], 
                                     [5,6,7,8]],[
                                     [-1,-2,-3,-4], [-5,-6,-7,-8]]]))
            
    def test_correct_label_batch(self):
        for _, y in self.dl:
            assert all(y == tensor([0,1]))
            
    def test_correct_batch_shape(self):
        for X, _ in self.dl:
            assert X.shape == (2,2,4)


    