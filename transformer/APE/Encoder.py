#encoding: utf-8

from torch import nn
from modules.base import Dropout, PositionalEmb

from transformer.Encoder import Encoder as EncoderBase
from transformer.Decoder import DecoderLayer as MSEncoderLayerBase

from utils.fmt.base import pad_id, parse_double_value_tuple

from math import sqrt

from cnfg.ihyp import *

class MSEncoderLayer(MSEncoderLayerBase):

	def forward(self, inpute, inputo, src_pad_mask=None, tgt_pad_mask=None):

		_inputo = self.layer_normer1(inputo)

		context = self.self_attn(_inputo, mask=tgt_pad_mask)

		if self.drop is not None:
			context = self.drop(context)

		context = context + (_inputo if self.norm_residual else inputo)

		_context = self.layer_normer2(context)
		_context_new = self.cross_attn(_context, inpute, mask=src_pad_mask)

		if self.drop is not None:
			_context_new = self.drop(_context_new)

		context = _context_new + (_context if self.norm_residual else context)

		context = self.ff(context)

		return context

class MSEncoder(nn.Module):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, emb_w=None, share_layer=False, disable_pemb=disable_std_pemb_encoder):

		super(MSEncoder, self).__init__()

		_ahsize = isize if ahsize is None else ahsize
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		self.drop = Dropout(dropout, inplace=True) if dropout > 0.0 else None

		self.wemb = nn.Embedding(nwd, isize, padding_idx=pad_id)
		if emb_w is not None:
			self.wemb.weight = emb_w

		self.pemb = None if disable_pemb else PositionalEmb(isize, xseql, 0, 0)
		if share_layer:
			_shared_layer = MSEncoderLayer(isize, _fhsize, dropout, attn_drop, num_head, _ahsize)
			self.nets = nn.ModuleList([_shared_layer for i in range(num_layer)])
		else:
			self.nets = nn.ModuleList([MSEncoderLayer(isize, _fhsize, dropout, attn_drop, num_head, _ahsize) for i in range(num_layer)])

		self.out_normer = nn.LayerNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters) if norm_output else None

	def forward(self, inpute, inputo, src_pad_mask=None, tgt_pad_mask=None):

		nquery = inputo.size(-1)

		out = self.wemb(inputo)

		out = out * sqrt(out.size(-1))
		if self.pemb is not None:
			out = out + self.pemb(inputo, expand=False)

		if self.drop is not None:
			out = self.drop(out)

		for net in self.nets:
			out = net(inpute, out, src_pad_mask=src_pad_mask, tgt_pad_mask=tgt_pad_mask)

		if self.out_normer is not None:
			out = self.out_normer(out)

		return out

class Encoder(nn.Module):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, global_emb=False, **kwargs):

		super(Encoder, self).__init__()

		nwd_src, nwd_tgt = parse_double_value_tuple(nwd)

		self.src_enc = EncoderBase(isize, nwd_src, num_layer, fhsize, dropout, attn_drop, num_head, xseql, ahsize, norm_output, **kwargs)

		emb_w = self.src_enc.wemb.weight if global_emb else None

		self.tgt_enc = MSEncoder(isize, nwd_tgt, num_layer, fhsize, dropout, attn_drop, num_head, xseql, ahsize, norm_output, emb_w, **kwargs)

	def forward(self, inpute, inputo, src_mask=None, tgt_mask=None):

		enc_src = self.src_enc(inpute, src_mask)

		return enc_src, self.tgt_enc(enc_src, inputo, src_mask, tgt_mask)

	def update_vocab(self, indices):

		_bind_emb = self.src_enc.wemb.weight.is_set_to(self.tgt_enc.wemb.weight)
		_swemb = nn.Embedding(len(indices), self.src_enc.wemb.weight.size(-1), padding_idx=pad_id)
		_twemb = nn.Embedding(len(indices), self.tgt_enc.wemb.weight.size(-1), padding_idx=pad_id)
		with torch.no_grad():
			_swemb.weight.copy_(self.src_enc.wemb.weight.index_select(0, indices))
		if _bind_emb:
			_twemb.weight = _swemb.weight
		else:
			with torch.no_grad():
				_twemb.weight.copy_(self.tgt_enc.wemb.weight.index_select(0, indices))
		self.src_enc.wemb, self.tgt_enc.wemb = _swemb, _twemb
