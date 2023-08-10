#encoding: utf-8

from torch import nn

class Encoder(nn.Module):

	def __init__(self, models, **kwargs):

		super(Encoder, self).__init__()
		self.nets = nn.ModuleList(models)

	# inputs: (bsize, seql)
	# mask: (bsize, 1, seql), generated with:
	#	mask = inputs.eq(pad_id).unsqueeze(1)

	def forward(self, *inputs, **kwargs):

		return [model(*inputs, **kwargs) for model in self.nets]
