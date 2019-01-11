#encoding: utf-8

from torch import nn

class Encoder(nn.Module):

	def __init__(self, models):

		super(Encoder, self).__init__()
		self.nets = models

	# inputs: (bsize, seql)
	# mask: (bsize, 1, seql), generated with:
	#	mask = inputs.eq(0).unsqueeze(1)

	def forward(self, inputs, mask=None):

		return [model(inputs, mask) for model in self.nets]
