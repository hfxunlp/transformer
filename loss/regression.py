#encoding: utf-8

import torch
from torch.nn.functional import kl_div
from torch.nn.modules.loss import _Loss

class KLRegLoss(_Loss):

	def __init__(self, reg_right=True, weight=None, bias=None, reduction="mean", **kwargs):

		super(KLRegLoss, self).__init__()
		self.weight, self.bias, self.reduction = weight, bias, reduction
		self.p_func = (lambda x: torch.cat((1.0 - x, x,), dim=-1)) if reg_right else (lambda x: torch.cat((x, 1.0 - x,), dim=-1))

	def forward(self, input, target, **kwargs):

		_target = target
		if self.bias is not None:
			_target = _target + self.bias
		if self.weight is not None:
			_target = _target * self.weight

		return kl_div(input, self.p_func(_target.unsqueeze(-1)), reduction=self.reduction)
