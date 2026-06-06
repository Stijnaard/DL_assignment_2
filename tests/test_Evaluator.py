from dl_assignment_2.modeling.evaluation import Evaluator
from dl_assignment_2.modeling.model import SimpleRNN
from dl_assignment_2.modeling.dataset import CustomDataset
from dl_assignment_2.data.dataSegment import DataSegment, SegmentInfo
from torch import Tensor, tensor
from torch.utils.data import DataLoader
from numpy import ndarray, array

class Test_stuff:
    x: ndarray = array(range(1,16)).reshape(3,5)
    y: ndarray = array(range(1,16)).reshape(3,5)

    s1: DataSegment = DataSegment(info=SegmentInfo(x, 123466, "rest",1))
    s2: DataSegment = DataSegment(info=SegmentInfo(y, 123466, "rest",1))

    ds: CustomDataset = CustomDataset(segments=(s1, s2))
    dl: DataLoader = DataLoader(ds, batch_size=1)
    rnn = SimpleRNN(3, 10, 4)
    eval = Evaluator(rnn, dl)

    def test_forward(self):
        assert self.eval.predictions.shape == (2,4)
