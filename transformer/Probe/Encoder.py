#encoding: utf-8

from transformer.Encoder import Encoder as EncoderBase

from math import sqrt

from cnfg.ihyp import *

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, num_layer_ana=0, **kwargs):

		super(Encoder, self).__init__(isize, nwd, num_layer, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, **kwargs)

		self.num_layer_ana = num_layer_ana

	def forward(self, inputs, mask=None, no_std_out=False):

		out = self.wemb(inputs)
		out = out * sqrt(out.size(-1))
		if self.pemb is not None:
			out = out + self.pemb(inputs, expand=False)

		if self.drop is not None:
			out = self.drop(out)

		lo = None

		if no_std_out:
			for net in self.nets[:self.num_layer_ana]:
				out = net(out, mask)
			if self.out_normer is not None:
				out = self.out_normer(out)
			return out
		else:
			for i, net in enumerate(self.nets):
				if self.num_layer_ana == i:
					lo = out
				out = net(out, mask)

			if self.out_normer is not None:
				out = self.out_normer(out)
				lo = out if lo is None else self.out_normer(lo)

			if lo is None:
				lo = out

			return out, lo
