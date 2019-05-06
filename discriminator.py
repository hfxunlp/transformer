#encoding: utf-8

from torch import nn
from modules.base import *
from math import sqrt
from transformer.Encoder import EncoderLayer

class MonoDiscriminator(nn.Module):

	def __init__(self, isize, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, num_head=8, xseql=512, ahsize=None, clip_value=None, use_pemb=True):

		super(MonoDiscriminator, self).__init__()

		_ahsize = isize if ahsize is None else ahsize

		_fhsize = _ahsize if fhsize is None else fhsize

		self.drop = nn.Dropout(dropout, inplace=True) if dropout > 0.0 else None

		self.pemb = PositionalEmb(isize, xseql, 0, 0) if use_pemb else None

		self.nets = nn.ModuleList([EncoderLayer(isize, _fhsize, dropout, attn_drop, num_head, _ahsize) for i in range(num_layer)])
		self.weight = nn.Parameter(torch.randn(isize))

		self.clip_value = clip_value

	def forward(self, inputs, mask=None):

		out = inputs if self.pemb is None else inputs * sqrt(inputs.size(-1)) + self.pemb(inputs, expand=False)

		if self.drop is not None:
			out = self.drop(out)

		for net in self.nets:
			out = net(out, mask)

		if mask is not None:
			bsize, _, seql = mask.size()
			out.masked_fill_(mask.view(bsize, seql, 1).expand_as(out), -1e32)

		out, _ = torch.max(out, dim = 1)
		out = torch.mv(out, self.weight)

		return out

	def fix_update(self):

		if self.clip_value is not None:
			for para in self.parameters():
				para.data.clamp_(- self.clip_value, self.clip_value)

class CompareLayer(nn.Module):

	# isize: input size
	# fhsize: hidden size of PositionwiseFeedForward
	# attn_drop: dropout for MultiHeadAttention
	# num_head: number of heads in MultiHeadAttention
	# ahsize: hidden size of MultiHeadAttention

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, num_head=8, ahsize=None):

		super(CompareLayer, self).__init__()

		_ahsize = isize if ahsize is None else ahsize

		_fhsize = _ahsize if fhsize is None else fhsize

		self.self_attn = MultiHeadAttn(isize, _ahsize, isize, num_head, dropout=attn_drop)
		self.cross_attn = MultiHeadAttn(isize, _ahsize, isize, num_head, dropout=attn_drop)

		self.ff = PositionwiseFF(isize, _fhsize, dropout)

		self.layer_normer1 = nn.LayerNorm(isize, eps=1e-06)
		self.layer_normer2 = nn.LayerNorm(isize, eps=1e-06)
		if dropout > 0:
			self.d1 = nn.Dropout(dropout, inplace=True)
			self.d2 = nn.Dropout(dropout, inplace=True)
		else:
			self.d1 = None
			self.d2 = None

	# inpute: encoded representation from encoder (bsize, seql, isize)
	# inputo: encoded representation from decoder (bsize, nquery, isize)
	# src_pad_mask: mask for given encoding source sentence (bsize, nquery, seql), see Encoder, expanded after generated with:
	#	src_pad_mask = input.eq(0).unsqueeze(1)
	# tgt_pad_mask: similar to src_pad_mask but for the target side

	def forward(self, inpute, inputo, src_pad_mask=None, tgt_pad_mask=None):

		_inputo = self.layer_normer1(inputo)

		context = self.self_attn(_inputo, _inputo, _inputo, mask=tgt_pad_mask)

		if self.d1 is not None:
			context = self.d1(context)

		context = context + inputo

		_context = self.layer_normer2(context)
		_context = self.cross_attn(_context, inpute, inpute, mask=src_pad_mask)

		if self.d2 is not None:
			_context = self.d2(_context)

		context = _context + context

		_context = self.ff(context)

		return _context + context

class PairDiscriminator(nn.Module):

	def __init__(self, isize, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, num_head=8, xseql=512, ahsize=None, clip_value=None, use_pemb=True):

		super(PairDiscriminator, self).__init__()

		_ahsize = isize if ahsize is None else ahsize

		_fhsize = _ahsize if fhsize is None else fhsize

		self.drop = nn.Dropout(dropout, inplace=True) if dropout > 0.0 else None

		self.pemb = PositionalEmb(isize, xseql, 0, 0) if use_pemb else None

		self.nets = nn.ModuleList([CompareLayer(isize, _fhsize, dropout, attn_drop, num_head, _ahsize) for i in range(num_layer)])
		self.weight = nn.Parameter(torch.randn(isize))

		self.clip_value = clip_value

	def forward(self, inpute, inputo, src_pad_mask=None, tgt_pad_mask=None):

		out = inputo if self.pemb is None else inputo * sqrt(inputo.size(-1)) + self.pemb(inputo, expand=False)

		if self.drop is not None:
			out = self.drop(out)

		for net in self.nets:
			out = net(inpute, out, src_pad_mask, tgt_pad_mask)

		if tgt_pad_mask is not None:
			bsize, _, nquery = tgt_pad_mask.size()
			out.masked_fill_(tgt_pad_mask.view(bsize, nquery, 1).expand_as(out), -1e32)

		out, _ = torch.max(out, dim = 1)
		out = torch.mv(out, self.weight)

		return out

	def fix_update(self):

		if self.clip_value is not None:
			for para in self.parameters():
				para.data.clamp_(- self.clip_value, self.clip_value)
