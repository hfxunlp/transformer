#encoding: utf-8

import torch
from torch import nn
from modules.base import *
from math import sqrt

from cnfg.vocab.base import pad_id

from cnfg.ihyp import *

# vocabulary:
#	<pad>:0
#	<sos>:1
#	<eos>:2
#	<unk>:3
#	...
# for the classier of the decoder, <sos> is omitted

class EncoderLayer(nn.Module):

	# isize: input size
	# fhsize: hidden size of PositionwiseFeedForward
	# attn_drop: dropout for MultiHeadAttention
	# num_head: number of heads in MultiHeadAttention
	# ahsize: hidden size of MultiHeadAttention
	# norm_residual: residue with layer normalized representation
	# k_rel_pos: window size (one side) of relative positional embeddings in self attention

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, num_head=8, ahsize=None, norm_residual=norm_residual_default, k_rel_pos=use_k_relative_position_encoder, max_bucket_distance=relative_position_max_bucket_distance_encoder, **kwargs):

		super(EncoderLayer, self).__init__()

		_ahsize = isize if ahsize is None else ahsize
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		self.attn = ResSelfAttn(isize, _ahsize, num_head=num_head, dropout=attn_drop, norm_residual=norm_residual, k_rel_pos=k_rel_pos, max_bucket_distance=max_bucket_distance)

		self.ff = PositionwiseFF(isize, hsize=_fhsize, dropout=dropout, norm_residual=norm_residual)

	# inputs: input of this layer (bsize, seql, isize)

	def forward(self, inputs, mask=None):

		context = self.attn(inputs, mask=mask)

		context = self.ff(context)

		return context

# Not used, keep this class to remind the EncoderLayer implementation before v0.3.5.
class NAWEncoderLayer(EncoderLayer):

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, num_head=8, ahsize=None, norm_residual=norm_residual_default, k_rel_pos=use_k_relative_position_encoder, max_bucket_distance=relative_position_max_bucket_distance_encoder, **kwargs):

		_ahsize = isize if ahsize is None else ahsize
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(NAWEncoderLayer, self).__init__(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, num_head=num_head, ahsize=_ahsize, norm_residual=norm_residual, k_rel_pos=k_rel_pos, max_bucket_distance=max_bucket_distance)

		#self.attn = SelfAttn(isize, _ahsize, isize, num_head=num_head, dropout=attn_drop, k_rel_pos=k_rel_pos, max_bucket_distance=max_bucket_distance)
		#self.ff = PositionwiseFF(isize, hsize=_fhsize, dropout=dropout, norm_residual=norm_residual)
		self.layer_normer, self.drop, self.norm_residual = self.attn.normer, self.attn.drop, self.attn.norm_residual
		self.attn = self.attn.net
		#self.layer_normer = nn.LayerNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)
		#self.drop = Dropout(dropout, inplace=True) if dropout > 0.0 else None
		#self.norm_residual = norm_residual

	def forward(self, inputs, mask=None):

		_inputs = self.layer_normer(inputs)
		context = self.attn(_inputs, mask=mask)

		if self.drop is not None:
			context = self.drop(context)

		context = context + (_inputs if self.norm_residual else inputs)

		context = self.ff(context)

		return context

class Encoder(nn.Module):

	# isize: size of word embedding
	# nwd: number of words
	# num_layer: number of encoder layers
	# fhsize: number of hidden units for PositionwiseFeedForward
	# attn_drop: dropout for MultiHeadAttention
	# num_head: number of heads in MultiHeadAttention
	# xseql: maxmimum length of sequence
	# ahsize: number of hidden units for MultiHeadAttention
	# share_layer: using one shared encoder layer
	# disable_pemb: disable the standard positional embedding, enable when use relative postional embeddings in self attention

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, share_layer=False, disable_pemb=disable_std_pemb_encoder, **kwargs):

		super(Encoder, self).__init__()

		_ahsize = isize if ahsize is None else ahsize
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		self.drop = Dropout(dropout, inplace=True) if dropout > 0.0 else None

		self.wemb = nn.Embedding(nwd, isize, padding_idx=pad_id)

		self.pemb = None if disable_pemb else PositionalEmb(isize, xseql, 0, 0)
		if share_layer:
			_shared_layer = EncoderLayer(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, num_head=num_head, ahsize=_ahsize)
			self.nets = nn.ModuleList([_shared_layer for i in range(num_layer)])
		else:
			self.nets = nn.ModuleList([EncoderLayer(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, num_head=num_head, ahsize=_ahsize) for i in range(num_layer)])

		self.out_normer = nn.LayerNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters) if norm_output else None

	# inputs: (bsize, seql)
	# mask: (bsize, 1, seql), generated with:
	#	mask = inputs.eq(0).unsqueeze(1)

	def forward(self, inputs, mask=None):

		out = self.wemb(inputs)
		out = out * sqrt(out.size(-1))
		if self.pemb is not None:
			out = out + self.pemb(inputs, expand=False)

		if self.drop is not None:
			out = self.drop(out)

		for net in self.nets:
			out = net(out, mask)

		return out if self.out_normer is None else self.out_normer(out)

	def load_base(self, base_encoder):

		self.drop = base_encoder.drop

		self.wemb = base_encoder.wemb

		self.pemb = base_encoder.pemb

		_nets = list(base_encoder.nets)

		self.nets = nn.ModuleList(_nets[:len(self.nets)] + list(self.nets[len(_nets):]))

		self.out_normer = None if self.out_normer is None else base_encoder.out_normer

	def update_vocab(self, indices):

		_wemb = nn.Embedding(len(indices), self.wemb.weight.size(-1), padding_idx=self.wemb.padding_idx)
		with torch.no_grad():
			_wemb.weight.copy_(self.wemb.weight.index_select(0, indices))
		self.wemb = _wemb

	def fix_init(self):

		if hasattr(self, "fix_load"):
			self.fix_load()
		#with torch.no_grad():
		#	self.wemb.weight[pad_id].zero_()
