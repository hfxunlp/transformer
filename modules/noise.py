#encoding: utf-8

import torch
from torch import nn

from modules.base import PositionwiseFF as PositionwiseFFBase

class GausNoiser(nn.Module):

	def __init__(self, power):

		super(GausNoiser, self).__init__()
		self.power = power

	# mask: (bsize, seql, 1), otherwise cannot multiply with inpute.size(-1)
	def forward(self, inpute, mask=None):

		if self.training:
			if mask is None:
				base_p = inpute.data.abs().mean() * self.power
			else:
				base_p = inpute.data.abs().masked_fill(mask, 0.0).sum() * (self.power / float((mask.numel() - mask.sum().item()) * inpute.size(-1)))

			return torch.randn(inpute.size(), dtype=inpute.dtype, device=inpute.device) * base_p + inpute

		return inpute

class UniNoiser(nn.Module):

	def __init__(self, power):

		super(UniNoiser, self).__init__()
		self.power = power

	def forward(self, inpute, mask=None):

		if self.training:
			if mask is None:
				base_p = inpute.data.abs().mean().item() * self.power
			else:
				base_p = inpute.data.abs().masked_fill(mask, 0.0).sum().item() / float((mask.numel() - mask.sum().item()) * inpute.size(-1)) * self.power

			return inpute.new_empty(inpute.size(), requires_grad=False).uniform_(- base_p, base_p) + inpute

		return inpute
