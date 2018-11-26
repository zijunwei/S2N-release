import torch
import torch.nn
from torch.autograd import Variable
torch.manual_seed(0)
x = Variable(torch.randn(3, 3), requires_grad=True)
# y = x.repeat(3, 3, 1, 1)
y = x.expand(3, 3, -1, -1)

conv = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3)
o1 = conv(y)
o2 = y.sum()
o = o1.sum() + o2
o.backward()
assert x.size() == x.grad.size()  # fails (2, 3, 4) != (1, 1, 2, 3, 4)