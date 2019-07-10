#encoding: utf-8

from math import sqrt
import torch
from torch import nn
from modules.base import GeLU_BERT
from modules.base import PositionwiseFF as PositionwiseFFBase

class GroupLinearCore(nn.Module):

	# isize: input dimension
	# osize: output dimension
	# ngroup: number of group
	# bias: enable bias or not

	def __init__(self, isize, osize, ngroup, bias=True):

		super(GroupLinearCore, self).__init__()

		self.ngroup = ngroup
		self.isize = isize // ngroup
		_osize = osize // ngroup

		self.weight = nn.Parameter(torch.Tensor(ngroup, self.isize, _osize).uniform_(- sqrt(2.0 / (self.isize + _osize)), sqrt(2.0 / (self.isize + _osize))))
		self.bias = nn.Parameter(torch.zeros(osize)) if bias else None

	# inputu: (bsize, isize)

	def forward(self, inputu):

		_bsize = inputu.size(0)
		out = inputu.view(_bsize, self.ngroup, self.isize).transpose(0, 1).bmm(self.weight).transpose(0, 1).contiguous().view(_bsize, -1)

		return out if self.bias is None else out + self.bias

class GroupLinear(nn.Module):

	# isize: input dimension
	# osize: output dimension
	# ngroup: number of group
	# bias: enable bias or not

	def __init__(self, isize, osize, ngroup, bias=True):

		super(GroupLinear, self).__init__()

		self.net = GroupLinearCore(isize, osize, ngroup, bias)

	# inputu: (..., isize)

	def forward(self, inputu):

		_size = list(inputu.size())
		_isize = _size[-1]
		_size[-1] = -1

		return self.net(inputu.view(-1, _isize)).view(_size)

class PositionwiseFF(PositionwiseFFBase):

	# isize: input dimension
	# hsize: hidden dimension

	def __init__(self, isize, hsize=None, dropout=0.0, norm_residue=False, use_GeLU=False, ngroup=16):

		_hsize = isize * 4 if hsize is None else hsize

		super(PositionwiseFF, self).__init__(isize, _hsize, dropout, norm_residue, use_GeLU)

		if dropout > 0.0:
			#self.nets = nn.ModuleList([nn.Linear(isize, _hsize), nn.Dropout(dropout, inplace=True), GeLU_BERT() if use_GeLU else nn.ReLU(inplace=True), GroupLinear(_hsize, _hsize, ngroup), nn.Dropout(dropout, inplace=True), GeLU_BERT() if use_GeLU else nn.ReLU(inplace=True), nn.Linear(_hsize, isize), nn.Dropout(dropout, inplace=True)])
			self.nets = nn.ModuleList([nn.Linear(isize, _hsize), nn.Dropout(dropout, inplace=True), GeLU_BERT() if use_GeLU else nn.ReLU(inplace=True), nn.Linear(_hsize, _hsize), nn.Dropout(dropout, inplace=True), GeLU_BERT() if use_GeLU else nn.ReLU(inplace=True), nn.Linear(_hsize, isize), nn.Dropout(dropout, inplace=True)])
		else:
			self.nets = nn.ModuleList([nn.Linear(isize, _hsize), GeLU_BERT() if use_GeLU else nn.ReLU(inplace=True), GroupLinear(_hsize, _hsize, ngroup), GeLU_BERT() if use_GeLU else nn.ReLU(inplace=True), nn.Linear(_hsize, isize)])
