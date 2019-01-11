#encoding: utf-8

from torch import nn
from modules import *
from math import sqrt

# vocabulary:
#	<pad>:0
#	<unk>:1
#	<eos>:2
#	<sos>:3
#	...
# for the classier of the decoder, <sos> is omitted

class EncoderLayer(nn.Module):

	# isize: input size
	# fhsize: hidden size of PositionwiseFeedForward
	# attn_drop: dropout for MultiHeadAttention
	# num_head: number of heads in MultiHeadAttention
	# ahsize: hidden size of MultiHeadAttention

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, num_head=8, ahsize=None):

		super(EncoderLayer, self).__init__()

		_ahsize = isize if ahsize is None else ahsize

		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		self.attn = MultiHeadAttn(isize, _ahsize, isize, num_head, dropout=attn_drop)

		self.ff = PositionwiseFF(isize, _fhsize, dropout, True)

		self.layer_normer = nn.LayerNorm(isize, eps=1e-06)

		self.drop = nn.Dropout(dropout) if dropout > 0.0 else None

	# inputs: input of this layer (bsize, seql, isize)

	def forward(self, inputs, mask=None):

		_inputs = self.layer_normer(inputs)
		context = self.attn(_inputs, _inputs, _inputs, mask=mask)

		if self.drop is not None:
			context = self.drop(context)

		context = context + inputs

		_context = self.ff(context)

		return _context + context

class Encoder(nn.Module):

	# isize: size of word embedding
	# nwd: number of words
	# num_layer: number of encoder layers
	# fhsize: number of hidden units for PositionwiseFeedForward
	# attn_drop: dropout for MultiHeadAttention
	# num_head: number of heads in MultiHeadAttention
	# xseql: maxmimum length of sequence
	# ahsize: number of hidden units for MultiHeadAttention

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, num_head=8, xseql=512, ahsize=None, norm_output=True):

		super(Encoder, self).__init__()

		_ahsize = isize if ahsize is None else ahsize

		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		self.drop = nn.Dropout(dropout) if dropout > 0.0 else None

		self.wemb = nn.Embedding(nwd, isize, padding_idx=0)

		self.pemb = PositionalEmb(isize, xseql, 0, 0)
		self.nets = nn.ModuleList([EncoderLayer(isize, _fhsize, dropout, attn_drop, num_head, _ahsize) for i in range(num_layer)])

		self.out_normer = nn.LayerNorm(isize, eps=1e-06) if norm_output else None

	# inputs: (bsize, seql)
	# mask: (bsize, 1, seql), generated with:
	#	mask = inputs.eq(0).unsqueeze(1)

	def forward(self, inputs, mask=None):

		bsize, seql = inputs.size()
		out = self.wemb(inputs)
		out = out * sqrt(out.size(-1)) + self.pemb(inputs, expand=False)

		if self.drop is not None:
			out = self.drop(out)

		for net in self.nets:
			out = net(out, mask)

		return out if self.out_normer is None else self.out_normer(out)
