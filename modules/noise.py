#encoding: utf-8

import torch
from torch import nn

from modules.base import ResSelfAttn as ResSelfAttnBase, ResCrossAttn as ResCrossAttnBase, PositionwiseFF as PositionwiseFFBase

from cnfg.ihyp import *

class GausNoiser(nn.Module):

	def __init__(self, power, inplace=False):

		super(GausNoiser, self).__init__()
		self.power, self.inplace = power, inplace

	# mask: (bsize, seql, 1), otherwise cannot multiply with inpute.size(-1)
	def forward(self, inpute, mask=None):

		if self.training:
			_noise = self.get_noise(inpute.detach(), mask=mask)

			return inpute.add_(_noise) if self.inplace else inpute.add(_noise)

		return inpute

	def get_noise(self, inpute, mask=None):

		if mask is None:
			base_p = inpute.abs().mean() * self.power
		else:
			base_p = inpute.masked_fill(mask, 0.0).norm(p=1) * (self.power / float((mask.numel() - mask.sum().item()) * inpute.size(-1)))

		return torch.randn(inpute.size(), dtype=inpute.dtype, device=inpute.device).mul_(base_p)

class UniNoiser(GausNoiser):

	def get_noise(self, inpute, mask=None):

		if mask is None:
			base_p = inpute.abs().mean().item() * self.power
		else:
			base_p = inpute.masked_fill(mask, 0.0).norm(p=1).item() / float((mask.numel() - mask.sum().item()) * inpute.size(-1)) * self.power

		return inpute.new_empty(inpute.size(), requires_grad=False).uniform_(-base_p, base_p)

class GausNoiserVec(GausNoiser):

	def __init__(self, power, dim=-1, inplace=False, eps=ieps_noise_default):

		super(GausNoiserVec, self).__init__(power, inplace=inplace)
		self.dim, self.eps = dim, eps

	def get_noise(self, inpute, mask=None):

		_noise = torch.randn(inpute.size(), dtype=inpute.dtype, device=inpute.device)
		base_p = inpute.norm(p=2, dim=self.dim, keepdim=True) / (_noise.norm(p=2, dim=self.dim, keepdim=True) + self.eps) * self.power

		return _noise.mul_(base_p)

class UniNoiserVec(GausNoiserVec):

	def get_noise(self, inpute, mask=None):

		_noise = inpute.new_empty(inpute.size(), requires_grad=False).uniform_(-1.0, 1.0)
		base_p = inpute.norm(p=2, dim=self.dim, keepdim=True) / (_noise.norm(p=2, dim=self.dim, keepdim=True) + self.eps) * self.power

		return _noise.mul_(base_p)

Noiser = UniNoiserVec

class ResSelfAttn(ResSelfAttnBase):

	def __init__(self, isize, hsize, num_head=8, dropout=0.0, norm_residual=norm_residual_default, power=None, custom_noiser=None, **kwargs):

		super(ResSelfAttn, self).__init__(isize, hsize, num_head=num_head, dropout=dropout, norm_residual=norm_residual, **kwargs)

		_noiser = Noiser if custom_noiser is None else custom_noiser
		self.noiser = None if power is None else _noiser(power, inplace=True)

	def forward(self, iQ, *inputs, noise_mask=None, **kwargs):

		_iQ = self.normer(iQ)

		if self.noiser is not None:
			_iQ = self.noiser(_iQ, noise_mask)

		outs = self.net(_iQ, *inputs, **kwargs)

		if isinstance(outs, tuple):
			_out = outs[0]

			if self.drop is not None:
				_out = self.drop(_out)

			return _out + (_iQ if self.norm_residual else iQ), *outs[1:]

		else:
			if self.drop is not None:
				outs = self.drop(outs)

			return outs + (_iQ if self.norm_residual else iQ)

class ResCrossAttn(ResCrossAttnBase):

	def __init__(self, isize, hsize, num_head=8, dropout=0.0, norm_residual=norm_residual_default, power=None, custom_noiser=None, **kwargs):

		super(ResCrossAttn, self).__init__(isize, hsize, num_head=num_head, dropout=dropout, norm_residual=norm_residual, **kwargs)

		_noiser = Noiser if custom_noiser is None else custom_noiser
		self.noiser = None if power is None else _noiser(power, inplace=True)

	def forward(self, iQ, iK, *inputs, noise_mask=None, **kwargs):

		_iQ = self.normer(iQ)

		if self.noiser is not None:
			_iQ = self.noiser(_iQ, noise_mask)

		outs = self.net(_iQ, iK, *inputs, **kwargs)

		if isinstance(outs, tuple):
			_out = outs[0]

			if self.drop is not None:
				_out = self.drop(_out)

			return _out + (_iQ if self.norm_residual else iQ), *outs[1:]

		else:
			if self.drop is not None:
				outs = self.drop(outs)

			return outs + (_iQ if self.norm_residual else iQ)

class PositionwiseFF(PositionwiseFFBase):

	def __init__(self, isize, power=None, custom_noiser=None, **kwargs):

		super(PositionwiseFF, self).__init__(isize, **kwargs)

		_noiser = Noiser if custom_noiser is None else custom_noiser
		self.noiser = None if power is None else _noiser(power, inplace=True)

	def forward(self, x, mask=None):

		_out = self.normer(x)
		if self.noiser is not None:
			_out = self.noiser(_out, mask)

		out = self.net(_out)

		out = out + (_out if self.norm_residual else x)

		return out
