#encoding: utf-8

import torch
from torch import nn
from modules.base import Linear

class GateResidual(nn.Module):

	# isize: input dimension

	def __init__(self, isize):

		super(GateResidual, self).__init__()

		self.net = nn.Sequential(Linear(isize * 2, isize), nn.Sigmoid())

	def forward(self, x1, x2):

		gate = self.net(torch.cat((x1, x2,), dim=-1))

		return x1 * gate + x2 * (1.0 - gate)
