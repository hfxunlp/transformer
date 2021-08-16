#encoding: utf-8

import torch
from torch.nn.functional import kl_div

from utils.base import eq_indexes

from loss.base import MultiLabelSmoothingLoss as MultiLabelSmoothingLossBase

class MultiLabelSmoothingLoss(MultiLabelSmoothingLossBase):

	def __init__(self, *inputs, **kwargs):

		super(MultiLabelSmoothingLoss, self).__init__(*inputs, **kwargs)
		self.register_buffer("weight", self.weight.squeeze(1))

	def forward(self, input, target, tinput):

		_rsize = list(input.size())
		_nclass = _rsize[-1]
		_mpvsize = [1 for i in range(len(_rsize))]
		_mpvsize[0] = _rsize[0]
		_mpvsize[-1] = _nclass
		_rsize[0] = 1
		_rsize[-1] = 1

		_input = input.view(-1, _nclass) if input.dim() > 2 else input
		_target = target.view(-1, 1)

		model_prob = self.weight.index_select(0, tinput).view(_mpvsize).repeat(*_rsize).view(-1, _nclass)
		model_prob.scatter_(1, _target, self.conf)

		if isinstance(self.ignore_index, (list, tuple)):
			model_prob.masked_fill_(eq_indexes(_target, self.ignore_index), 0.0)
		elif self.ignore_index >= 0:
			model_prob.masked_fill_(_target == self.ignore_index, 0.0)

		rs = kl_div(_input, model_prob, reduction=self.reduction)

		return rs.view(input.size()) if self.reduction == 'none' and target.dim() > 1 else rs
