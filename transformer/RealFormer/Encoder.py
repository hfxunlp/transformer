#encoding: utf-8

from torch import nn
from math import sqrt

from modules.attn.res import SelfAttn

from transformer.Encoder import EncoderLayer as EncoderLayerBase
from transformer.Encoder import Encoder as EncoderBase

from cnfg.ihyp import *

class EncoderLayer(EncoderLayerBase):

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, num_head=8, ahsize=None, norm_residual=norm_residual_default, k_rel_pos=use_k_relative_position_encoder, **kwargs):

		_ahsize = isize if ahsize is None else ahsize
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(EncoderLayer, self).__init__(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, num_head=num_head, ahsize=_ahsize, norm_residual=norm_residual, k_rel_pos=k_rel_pos, **kwargs)

		self.attn = SelfAttn(isize, _ahsize, isize, num_head=num_head, dropout=attn_drop, k_rel_pos=k_rel_pos)

	def forward(self, inputs, mask=None, resin=None):

		_inputs = self.layer_normer(inputs)
		context, resout = self.attn(_inputs, mask=mask, resin=resin)

		if self.drop is not None:
			context = self.drop(context)

		context = context + (_inputs if self.norm_residual else inputs)

		context = self.ff(context)

		return context, resout

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, share_layer=False, **kwargs):

		_ahsize = isize if ahsize is None else ahsize
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Encoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, share_layer=share_layer, **kwargs)

		if share_layer:
			_shared_layer = EncoderLayer(isize, _fhsize, dropout, attn_drop, num_head, _ahsize)
			self.nets = nn.ModuleList([_shared_layer for i in range(num_layer)])
		else:
			self.nets = nn.ModuleList([EncoderLayer(isize, _fhsize, dropout, attn_drop, num_head, _ahsize) for i in range(num_layer)])

	def forward(self, inputs, mask=None):

		out = self.wemb(inputs)
		out = out * sqrt(out.size(-1))
		if self.pemb is not None:
			out = out + self.pemb(inputs, expand=False)

		if self.drop is not None:
			out = self.drop(out)

		attnres = None
		for net in self.nets:
			out, attnres = net(out, mask=mask, resin=attnres)

		return out if self.out_normer is None else self.out_normer(out)
