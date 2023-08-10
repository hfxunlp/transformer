#encoding: utf-8

from torch import nn

from modules.base import PositionwiseFF as PositionwiseFFBase, ResCrossAttn as ResCrossAttnBase, ResSelfAttn as ResSelfAttnBase
from modules.dropout import Dropout

from cnfg.ihyp import *

class SDU(nn.Sequential):

	def __init__(self, isize, dropout=0.0, **kwargs):

		super(SDU, self).__init__(nn.Linear(isize, isize + isize), nn.GLU())

		if dropout > 0.0:
			self.append(Dropout(dropout, inplace=True))

class PositionwiseFF(PositionwiseFFBase):

	def __init__(self, isize, hsize=None, dropout=0.0, act_drop=None, norm_residual=norm_residual_default, **kwargs):

		super(PositionwiseFF, self).__init__(isize, hsize=hsize, dropout=dropout, act_drop=act_drop, norm_residual=norm_residual, **kwargs)

		self.sdu = SDU(isize, dropout=dropout)

	def forward(self, x, **kwargs):

		_out = self.normer(x)

		out = self.net(_out) + self.sdu(_out)

		out = out + (_out if self.norm_residual else x)

		return out

class ResSelfAttn(ResSelfAttnBase):

	def __init__(self, isize, hsize, num_head=8, dropout=0.0, norm_residual=norm_residual_default, **kwargs):

		super(ResSelfAttn, self).__init__(isize, hsize, num_head=num_head, dropout=dropout, norm_residual=norm_residual, **kwargs)

		self.sdu = SDU(isize, dropout=dropout)

	def forward(self, iQ, *inputs, **kwargs):

		_iQ = self.normer(iQ)

		outs = self.net(_iQ, *inputs, **kwargs)

		if isinstance(outs, tuple):
			_out = outs[0]

			if self.drop is not None:
				_out = self.drop(_out)

			return _out + self.sdu(_iQ) + (_iQ if self.norm_residual else iQ), *outs[1:]

		else:
			if self.drop is not None:
				outs = self.drop(outs)

			return outs + self.sdu(_iQ) + (_iQ if self.norm_residual else iQ)

class ResCrossAttn(ResCrossAttnBase):

	def __init__(self, isize, hsize, num_head=8, dropout=0.0, norm_residual=norm_residual_default, **kwargs):

		super(ResCrossAttn, self).__init__(isize, hsize, num_head=num_head, dropout=dropout, norm_residual=norm_residual, **kwargs)

		self.sdu = SDU(isize, dropout=dropout)

	def forward(self, iQ, iK, *inputs, **kwargs):

		_iQ = self.normer(iQ)

		outs = self.net(_iQ, iK, *inputs, **kwargs)

		if isinstance(outs, tuple):
			_out = outs[0]

			if self.drop is not None:
				_out = self.drop(_out)

			return _out + self.sdu(_iQ) + (_iQ if self.norm_residual else iQ), *outs[1:]

		else:
			if self.drop is not None:
				outs = self.drop(outs)

			return outs + self.sdu(_iQ) + (_iQ if self.norm_residual else iQ)
