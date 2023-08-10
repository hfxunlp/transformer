#encoding: utf-8

import torch
from math import sqrt
from numbers import Integral
from torch import nn
from torch.nn import functional as nnFunc

from modules.base import PositionwiseFF as PositionwiseFFBase, ResCrossAttn as ResCrossAttnBase, ResSelfAttn as ResSelfAttnBase
from utils.torch.comp import torch_no_grad

from cnfg.ihyp import *

class MBLinear(nn.Linear):

	def __init__(self, in_features, out_features, nbias, bias=True, **kwargs):

		super(MBLinear, self).__init__(in_features, out_features, bias=False)

		if bias:
			self.bias = nn.Parameter(torch.zeros(nbias, out_features))

	def forward(self, x, taskid, **kwargs):

		return nnFunc.linear(x, self.weight, None if self.bias is None else self.bias[taskid])

	def fix_init(self):

		if self.bias is not None:
			with torch_no_grad():
				self.bias.zero_()

class MWLinear(MBLinear):

	def __init__(self, in_features, out_features, nbias, bias=True, **kwargs):

		super(MWLinear, self).__init__(in_features, out_features, nbias, bias=False)

		self.weight = nn.Parameter(torch.Tensor(nbias, out_features, in_features).uniform_(- sqrt(1.0 / in_features), sqrt(1.0 / in_features)))

	def forward(self, x, taskid, **kwargs):

		return nnFunc.linear(x, self.weight[taskid], None if self.bias is None else self.bias[taskid])

	def fix_init(self):

		_isize = self.weight.size(-1)
		with torch_no_grad():
			self.weight.data.uniform_(- sqrt(1.0 / _isize), sqrt(1.0 / _isize))
		super(MWLinear, self).fix_init()

class LayerNorm(nn.LayerNorm):

	def __init__(self, normalized_shape, ntask=None, eps=1e-5, elementwise_affine=True, **kwargs):

		if isinstance(normalized_shape, Integral):
			normalized_shape = (ntask, normalized_shape,)
		else:
			normalized_shape = tuple([ntask, *normalized_shape])

		super(LayerNorm, self).__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine, **kwargs)

		self.normalized_shape = self.normalized_shape[1:]

	def forward(self, input, taskid=None, **kwargs):

		return nnFunc.layer_norm(input, self.normalized_shape, None if self.weight is None else self.weight[taskid], None if self.bias is None else self.bias[taskid], self.eps)

class ResSelfAttn(ResSelfAttnBase):

	def __init__(self, isize, hsize, num_head=8, dropout=0.0, norm_residual=norm_residual_default, ntask=None, **kwargs):

		super(ResSelfAttn, self).__init__(isize, hsize, num_head=num_head, dropout=dropout, norm_residual=norm_residual, **kwargs)

		self.normer = LayerNorm(isize, ntask=ntask, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)

	def forward(self, iQ, *inputs, taskid=None, **kwargs):

		_iQ = self.normer(iQ, taskid=taskid)

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

	def __init__(self, isize, hsize, num_head=8, dropout=0.0, norm_residual=norm_residual_default, ntask=None, **kwargs):

		super(ResCrossAttn, self).__init__(isize, hsize, num_head=num_head, dropout=dropout, norm_residual=norm_residual, **kwargs)

		self.normer = LayerNorm(isize, ntask=ntask, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)

	def forward(self, iQ, iK, *inputs, taskid=None, **kwargs):

		_iQ = self.normer(iQ, taskid=taskid)

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

	def __init__(self, isize, hsize=None, dropout=0.0, act_drop=None, norm_residual=norm_residual_default, ntask=None, **kwargs):

		super(PositionwiseFF, self).__init__(isize, hsize=hsize, dropout=dropout, act_drop=act_drop, norm_residual=norm_residual, **kwargs)

		self.normer = LayerNorm(isize, ntask=ntask, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)

	def forward(self, x, taskid=None, **kwargs):

		_out = self.normer(x, taskid=taskid)

		out = self.net(_out)

		out = out + (_out if self.norm_residual else x)

		return out
