#encoding: utf-8

import torch
from torch import nn

from modules.base import PositionwiseFF as PositionwiseFFBase

from cnfg.ihyp import *

class GausNoiser(nn.Module):

	def __init__(self, power):

		super(GausNoiser, self).__init__()
		self.power = power

	# mask: (bsize, seql, 1), otherwise cannot multiply with inpute.size(-1)
	def forward(self, inpute, mask=None):

		if self.training:
			if mask is None:
				base_p = inpute.detach().abs().mean() * self.power
			else:
				base_p = inpute.detach().masked_fill(mask, 0.0).norm(p=1) * (self.power / float((mask.numel() - mask.sum().item()) * inpute.size(-1)))

			return torch.randn(inpute.size(), dtype=inpute.dtype, device=inpute.device) * base_p + inpute

		return inpute

class UniNoiser(nn.Module):

	def __init__(self, power):

		super(UniNoiser, self).__init__()
		self.power = power

	def forward(self, inpute, mask=None):

		if self.training:
			if mask is None:
				base_p = inpute.detach().abs().mean().item() * self.power
			else:
				base_p = inpute.detach().masked_fill(mask, 0.0).norm(p=1).item() / float((mask.numel() - mask.sum().item()) * inpute.size(-1)) * self.power

			return inpute.new_empty(inpute.size(), requires_grad=False).uniform_(-base_p, base_p) + inpute

		return inpute

class GausNoiserVec(nn.Module):

	def __init__(self, power, dim=-1, eps=ieps_noise_default):

		super(GausNoiserVec, self).__init__()
		self.power = power
		self.dim = dim
		self.eps = eps

	def forward(self, inpute, mask=None):

		if self.training:
			_noise = torch.randn(inpute.size(), dtype=inpute.dtype, device=inpute.device)
			base_p = inpute.detach().norm(p=2, dim=self.dim, keepdim=True) / (_noise.norm(p=2, dim=self.dim, keepdim=True) + self.eps) * self.power

			return _noise * base_p + inpute

		return inpute

class UniNoiserVec(nn.Module):

	def __init__(self, power, dim=-1, eps=ieps_noise_default):

		super(UniNoiserVec, self).__init__()
		self.power = power
		self.dim = dim
		self.eps = eps

	def forward(self, inpute, mask=None):

		if self.training:
			_noise = inpute.new_empty(inpute.size(), requires_grad=False).uniform_(-1.0, 1.0)
			base_p = inpute.detach().norm(p=2, dim=self.dim, keepdim=True) / (_noise.norm(p=2, dim=self.dim, keepdim=True) + self.eps) * self.power

			return _noise * base_p + inpute

		return inpute

Noiser = UniNoiserVec

class PositionwiseFF(PositionwiseFFBase):

	def __init__(self, isize, power=None, **kwargs):

		super(PositionwiseFF, self).__init__(isize, **kwargs)

		self.noiser = None if power is None else Noiser(power)

	def forward(self, x, mask=None):

		_out = self.normer(x)
		if self.noiser is not None:
			_out = self.noiser(_out, mask)

		out = self.net(_out)

		out = out + (_out if self.norm_residual else x)

		return out
