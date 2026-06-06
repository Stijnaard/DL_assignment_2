from dl_assignment_2.modeling.model import SimpleRNN
from dl_assignment_2.data.dataSegment import DataSegment, SegmentInfo
from dl_assignment_2.modeling.dataset import CustomDataset

from torch.utils.data import DataLoader
from numpy import ndarray, array
from torch import Tensor, ones, nn, optim

class Test_output_shapes:
    rnn = SimpleRNN(10,20,4)
    def test_one(self):
        x: Tensor = ones((3, 10))   # BATCH, SEQ LEN, INPUT DIM
        y = self.rnn(x)
        assert y.shape == (4,)
        
    def test_batch(self):
        x: Tensor = ones(5,3, 10)
        y = self.rnn(x)
        assert y.shape == (5, 4) # always adds extra dimension at axis=0
        
class Test_Predicion_Sanity:
    rnn = SimpleRNN(10,20,4)
    
    def test_one(self):
        x: Tensor = ones((1,3, 10))   # BATCH, SEQ LEN, INPUT DIM
        y = self.rnn.predict(x)
        logits = self.rnn(x)
        
    def test_batch(self):
        x: Tensor = ones((2, 3, 10))   # BATCH, SEQ LEN, INPUT DIM
        y = self.rnn.predict(x)
        logits = self.rnn(x)

class Test_computation_from_dataset:
    m1: ndarray = array([[1,2,3,4], [5,6,7,8]])
    m2: ndarray = m1*-1

    s1: DataSegment = DataSegment(info=SegmentInfo(m1, 123456, "rest", 1))
    s2: DataSegment = DataSegment(info=SegmentInfo(m2, 123456, "task_motor", 1))

    ds: CustomDataset = CustomDataset((s1, s2))
    batch_dl: DataLoader = DataLoader(ds, 2)
    dl: DataLoader = DataLoader(ds, 1)
    
    def test_single_computation_from_loader(self):
        n_sens: int = 2
        hidden_dim: int = 10
        output_dim: int = 4
        batch_size: int = 1
        
        model: SimpleRNN = SimpleRNN(n_sens, hidden_dim, output_dim)
        
        for x, _ in self.dl:
            y = model(x)
            
            assert y.shape == (batch_size, output_dim)
            
            break
            
    def test_batch_computation_from_loader(self):
        n_sens: int = 2
        hidden_dim: int = 10
        output_dim: int = 4
        batch_size: int = 2
        
        model: SimpleRNN = SimpleRNN(n_sens, hidden_dim, output_dim)
        
        for x, _ in self.batch_dl:
            y = model(x)
            
            assert y.shape == (batch_size, output_dim)
            
            break
        
    def test_forward_pass_w_loss(self):
        n_sens          : int = 2
        hidden_dim      : int = 10
        output_dim      : int = 4
        learning_rate   : float = 0.05
        
        model: SimpleRNN = SimpleRNN(n_sens, hidden_dim, output_dim)
        loss_func = nn.CrossEntropyLoss()
        
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        
        for x, y in self.dl:    # x.shape = (1,1,4) = (batch, n_in, seq_len)
            pred = model(x)     # y.shape = [1[]
            loss = loss_func(pred, y)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            break

    def test_forward_pass_batch_loss(self):
        n_sens          : int = 2
        hidden_dim      : int = 10
        output_dim      : int = 4
        learning_rate   : float = 0.05
        
        model: SimpleRNN = SimpleRNN(n_sens, hidden_dim, output_dim)
        loss_func = nn.CrossEntropyLoss()
        
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        
        for x, y in self.batch_dl:    # x.shape = (1,1,4) = (batch, n_in, seq_len)
            pred = model(x)     # y.shape = [1[]
            loss = loss_func(pred, y)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            break
        
    def test_learning(self):
        n_sens          : int = 2
        hidden_dim      : int = 10
        output_dim      : int = 4
        learning_rate   : float = 0.05
        
        model: SimpleRNN = SimpleRNN(n_sens, hidden_dim, output_dim)
        loss_func = nn.CrossEntropyLoss()
        
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        
        for x, y in self.batch_dl:    # x.shape = (1,1,4) = (batch, n_in, seq_len)
            pred = model(x)     # y.shape = [1[]
            loss = loss_func(pred, y)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            new_pred = model(x)
            new_loss = loss_func(new_pred, y)
            assert new_loss < loss
            
            break