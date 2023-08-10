#encoding: utf-8

import torch
from math import sqrt
from torch import nn

from modules.TA import PositionwiseFF, ResSelfAttn
from modules.base import Dropout
from transformer.Encoder import Encoder as EncoderBase, EncoderLayer as EncoderLayerBase
from utils.fmt.parser import parse_none

from cnfg.ihyp import *

class EncoderLayer(EncoderLayerBase):

	# isize: input size
	# fhsize: hidden size of PositionwiseFeedForward
	# attn_drop: dropout for MultiHeadAttention
	# num_head: number of heads in MultiHeadAttention
	# ahsize: hidden size of MultiHeadAttention

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, ahsize=None, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(EncoderLayer, self).__init__(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize, **kwargs)

		self.attn = ResSelfAttn(isize, _ahsize, num_head=num_head, dropout=attn_drop)
		self.ff = PositionwiseFF(isize, hsize=_fhsize, dropout=dropout, act_drop=act_drop)

class Encoder(EncoderBase):

	# isize: size of word embedding
	# nwd: number of words
	# num_layer: number of encoder layers
	# fhsize: number of hidden units for PositionwiseFeedForward
	# attn_drop: dropout for MultiHeadAttention
	# num_head: number of heads in MultiHeadAttention
	# xseql: maxmimum length of sequence
	# ahsize: number of hidden units for MultiHeadAttention

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, share_layer=False, num_layer_dec=6, **kwargs):

		_ahsize = parse_none(ahsize, isize)

		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Encoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, share_layer=share_layer, **kwargs)

		if share_layer:
			_shared_layer = EncoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize)
			self.nets = nn.ModuleList([_shared_layer for i in range(num_layer)])
		else:
			self.nets = nn.ModuleList([EncoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize) for i in range(num_layer)])

		self.tattn_w = nn.Parameter(torch.Tensor(num_layer + 1, num_layer_dec).uniform_(- sqrt(1.0 / (num_layer + 1)), sqrt(1.0 / (num_layer + 1))))
		self.tattn_drop = Dropout(dropout) if dropout > 0.0 else None

	# inputs: (bsize, seql)
	# mask: (bsize, 1, seql), generated with:
	#	mask = inputs.eq(pad_id).unsqueeze(1)

	def forward(self, inputs, mask=None, **kwargs):

		bsize, seql = inputs.size()
		out = self.wemb(inputs)
		if self.pemb is not None:
			out = self.pemb(inputs, expand=False).add(out, alpha=sqrt(out.size(-1)))

		if self.drop is not None:
			out = self.drop(out)

		out = self.out_normer(out)
		outs = [out]

		for net in self.nets:
			out = net(out, mask)
			outs.append(out)

		out = torch.stack(outs, -1)

		osize = out.size()
		out = out.view(-1, osize[-1]).mm(self.tattn_w.softmax(dim=0) if self.tattn_drop is None else self.tattn_drop(self.tattn_w).softmax(dim=0))
		osize = list(osize)
		osize[-1] = -1

		return out.view(osize)
