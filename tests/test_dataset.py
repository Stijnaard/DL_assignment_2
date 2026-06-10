from dl_assignment_2.modeling.dataset import CustomDataset
from dl_assignment_2.data.dataSegment import DataSegment, SegmentInfo

from numpy import ndarray, array
from torch.utils.data import DataLoader
from torch import tensor, all, transpose

class Base_TestDataset:
    m1: ndarray = array([[1,2,3,4], [5,6,7,8]])
    m2: ndarray = m1*-1

    s1: DataSegment = DataSegment(info=SegmentInfo(m1, 123456, "rest", 1))
    s2: DataSegment = DataSegment(info=SegmentInfo(m2, 123456, "task_motor", 1))

    ds: CustomDataset = CustomDataset((s1, s2))
    dl: DataLoader = DataLoader(ds, 2)

class Test_Dataset_single_instances(Base_TestDataset):
    def test_correct_feature_matrix(self):
        x, _ = self.ds.__getitem__(0)
        assert all(x == tensor([[1,5], [2,6], [3,7], [4,8]]))
        

    def test_correct_label(self):
        _, y = self.ds.__getitem__(0)
        assert y == 0   # 0 = rest (moves alphabetically)

    def test_correct_length(self):
        assert len(self.ds) == 2

class Test_Dataset_Batches(Base_TestDataset):
    def test_correct_feature_batch(self):
        for X, _ in self.dl:
            assert all(X == tensor([[[1,5], [2,6], [3,7], [4,8]], [[-1,-5], [-2,-6], [-3,-7], [-4,-8]]]))
            break
            
    def test_correct_label_batch(self):
        for _, y in self.dl:
            assert all(y == tensor([0,1]))
            
    def test_correct_batch_shape(self):
        for X, _ in self.dl:
            assert X.shape == (2,4,2)
    
class Test_Preps_Data_Correctly:
    sens1 = [1,2,3,4,5]
    sens2 = [5,6,7,8,5]
    sens3 = [9,10,11,12,5]
    
    m1 = array([sens1,sens2,sens3])
    s1 = DataSegment(info=SegmentInfo(m1, 123456, "rest", 1))
    
    ds = CustomDataset([s1])
    dl = DataLoader(ds, batch_size=1)
    
    def test_correct_dimensionality_non_batched(self):
        for x, y in self.dl:
            batch: int = 1
            seq_len: int = 5
            input_dim: int = 3
            assert x.shape == (batch, seq_len, input_dim)
    
    
    


    