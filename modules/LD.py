#encoding: utf-8

import torch
from torch import nn

from modules.act import Custom_Act
from modules.base import Dropout, Linear, Scorer

from cnfg.ihyp import *

class ATTNCombiner(nn.Module):

	def __init__(self, isize, hsize=None, dropout=0.0, custom_act=use_adv_act_default, **kwargs):

		super(ATTNCombiner, self).__init__()

		_hsize = isize * 4 if hsize is None else hsize

		self.net = nn.Sequential(Linear(isize * 2, _hsize), Dropout(dropout, inplace=True), Custom_Act() if custom_act else nn.Sigmoid(), Scorer(_hsize), nn.Sigmoid()) if dropout > 0.0 else nn.Sequential(Linear(isize * 2, _hsize), Custom_Act() if custom_act else nn.Sigmoid(), Scorer(_hsize), nn.Sigmoid())

	def forward(self, input1, input2, mask=None, **kwargs):

		scores = self.net(torch.cat((input1.expand_as(input2), input2,), dim=-1))

		_seql = input2.size(-2)
		if mask is not None:
			_tm = mask.sum(-2, keepdim=True)
			_nele = (_seql - _tm).masked_fill(_tm.eq(_seql), 1).to(scores, non_blocking=True)
			scores = scores / _nele
		else:
			scores = scores / _seql
		scores = scores.masked_fill(mask, 0.0)

		out = scores.transpose(1, 2).bmm(input2) + (1.0 - scores.sum(dim=-2, keepdim=True)) * input1

		return out

class DATTNCombiner(nn.Module):

	def __init__(self, isize, hsize=None, dropout=0.0, custom_act=use_adv_act_default, **kwargs):

		super(DATTNCombiner, self).__init__()

		_hsize = isize * 4 if hsize is None else hsize

		self.net = nn.Sequential(Linear(isize * 2, _hsize), Dropout(dropout, inplace=True), Custom_Act() if custom_act else nn.Sigmoid(), Scorer(_hsize, bias=False)) if dropout > 0.0 else nn.Sequential(Linear(isize * 2, _hsize), Custom_Act() if custom_act else nn.Sigmoid(), Scorer(_hsize, bias=False))

	# input1: (bsize, 1, isize)
	# input2: (bsize, seql, isize)
	# mask: (bsize, seql, 1)
	def forward(self, input1, input2, mask=None, **kwargs):

		# scores: (bsize, seql, 1)
		scores = self.net(torch.cat((input1.expand_as(input2), input2,), dim=-1))

		_seql = input2.size(-2)
		if mask is not None:
			# using math.inf as inf_default will lead to nan after softmax in case a sequence is full of <pad> tokens, advice: using the other values as inf_default, like 1e9.
			scores = scores.masked_fill(mask, -inf_default)

		scores = scores.softmax(dim=-2)

		# out: (bsize, 1, isize)
		out = scores.transpose(1, 2).bmm(input2)

		return out
