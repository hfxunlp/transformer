#encoding: utf-8

import torch
import torch.nn.functional as F
from loss import LabelSmoothingLoss

lossf=LabelSmoothingLoss(8, label_smoothing=0.1, ignore_index=0, reduction='none', forbidden_index=3)
target=torch.ones(5,1).long()
target.data[0]=0
target.data[1]=1
target.data[2]=2
target.data[3]=4
td=torch.randn(5,8)
#td.narrow(1, 3, 1).fill_(-1e32)
td.requires_grad_(True)
print(td)
output=F.log_softmax(td, -1)
print(output)
cost=lossf(output, target)
print(cost)
cost.sum().backward()
print(output.grad)
print(td.grad)
