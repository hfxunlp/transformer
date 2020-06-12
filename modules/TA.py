#encoding: utf-8

from modules.base import PositionwiseFF as PositionwiseFFBase

from cnfg.ihyp import *

class PositionwiseFF(PositionwiseFFBase):

	# isize: input dimension
	# hsize: hidden dimension

	def __init__(self, isize, norm_residual=norm_residual_default, **kwargs):

		super(PositionwiseFF, self).__init__(isize, norm_residual=False, **kwargs)

	def forward(self, x):

		return self.normer(self.net(x) + x)
