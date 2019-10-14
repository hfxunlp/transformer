#encoding: utf-8

from modules.base import PositionwiseFF as PositionwiseFFBase

class PositionwiseFF(PositionwiseFFBase):

	# isize: input dimension
	# hsize: hidden dimension

	def __init__(self, isize, hsize=None, dropout=0.0, use_GeLU=False):

		super(PositionwiseFF, self).__init__(isize, hsize, dropout, False, use_GeLU)

	def forward(self, x):

		return self.normer(self.net(x) + x)
