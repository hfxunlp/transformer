#encoding: utf-8

from math import sqrt
from torch import nn

from modules.base import Dropout, PositionalEmb
from transformer.Decoder import DecoderLayer as MSEncoderLayerBase
from transformer.Encoder import Encoder as EncoderBase
from utils.fmt.base import parse_double_value_tuple
from utils.fmt.parser import parse_none
from utils.torch.comp import torch_no_grad

from cnfg.ihyp import *
from cnfg.vocab.base import pad_id

class MSEncoderLayer(MSEncoderLayerBase):

	def forward(self, inpute, inputo, src_pad_mask=None, tgt_pad_mask=None, **kwargs):

		context = self.self_attn(inputo, mask=tgt_pad_mask)

		context = self.cross_attn(context, inpute, mask=src_pad_mask)

		context = self.ff(context)

		return context

class MSEncoder(nn.Module):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, emb_w=None, share_layer=False, disable_pemb=disable_std_pemb_encoder, **kwargs):

		super(MSEncoder, self).__init__()

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		self.drop = Dropout(dropout, inplace=True) if dropout > 0.0 else None

		self.wemb = nn.Embedding(nwd, isize, padding_idx=pad_id)
		if emb_w is not None:
			self.wemb.weight = emb_w

		self.pemb = None if disable_pemb else PositionalEmb(isize, xseql, 0, 0)
		if share_layer:
			_shared_layer = MSEncoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize)
			self.nets = nn.ModuleList([_shared_layer for i in range(num_layer)])
		else:
			self.nets = nn.ModuleList([MSEncoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize) for i in range(num_layer)])

		self.out_normer = nn.LayerNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters) if norm_output else None

	def forward(self, inpute, inputo, src_pad_mask=None, tgt_pad_mask=None, **kwargs):

		nquery = inputo.size(-1)

		out = self.wemb(inputo)
		if self.pemb is not None:
			out = self.pemb(inputo, expand=False).add(out, alpha=sqrt(out.size(-1)))

		if self.drop is not None:
			out = self.drop(out)

		for net in self.nets:
			out = net(inpute, out, src_pad_mask=src_pad_mask, tgt_pad_mask=tgt_pad_mask)

		if self.out_normer is not None:
			out = self.out_normer(out)

		return out

class Encoder(nn.Module):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, global_emb=False, **kwargs):

		super(Encoder, self).__init__()

		nwd_src, nwd_tgt = parse_double_value_tuple(nwd)

		self.src_enc = EncoderBase(isize, nwd_src, num_layer, fhsize, dropout, attn_drop, act_drop, num_head, xseql, ahsize, norm_output, **kwargs)

		emb_w = self.src_enc.wemb.weight if global_emb else None

		self.tgt_enc = MSEncoder(isize, nwd_tgt, num_layer, fhsize, dropout, attn_drop, act_drop, num_head, xseql, ahsize, norm_output, emb_w, **kwargs)

	def forward(self, inpute, inputo, src_mask=None, tgt_mask=None, **kwargs):

		enc_src = self.src_enc(inpute, src_mask)

		return enc_src, self.tgt_enc(enc_src, inputo, src_mask, tgt_mask)

	def get_embedding_weight(self):

		return self.enc_src.get_embedding_weight()

	def update_vocab(self, indices):

		_bind_emb = self.src_enc.wemb.weight.is_set_to(self.tgt_enc.wemb.weight)
		_ = self.src_enc.update_vocab(indices)
		if _bind_emb:
			self.tgt_enc.wemb.weight = _

		return _
