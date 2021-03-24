#encoding: utf-8

import torch
from torch import nn

from modules.base import Linear, Dropout
from modules.attn.rap import CrossAttn

from math import sqrt

from transformer.Decoder import DecoderLayer as DecoderLayerBase
from transformer.Decoder import Decoder as DecoderBase

from cnfg.ihyp import *

class DecoderLayer(DecoderLayerBase):

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, num_head=8, ahsize=None, **kwargs):

		_ahsize = isize if ahsize is None else ahsize

		super(DecoderLayer, self).__init__(isize, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, num_head=num_head, ahsize=_ahsize, **kwargs)

		self.cross_attn = CrossAttn(isize, _ahsize, isize, num_head, dropout=attn_drop)

	def forward(self, inpute, inputo, src_pad_mask=None, tgt_pad_mask=None, query_unit=None, compute_ffn=True):

		if query_unit is None:
			_inputo = self.layer_normer1(inputo)

			context = self.self_attn(_inputo, mask=tgt_pad_mask)

			if self.drop is not None:
				context = self.drop(context)

			context = context + (_inputo if self.norm_residual else inputo)

		else:
			_query_unit = self.layer_normer1(query_unit)

			context, states_return = self.self_attn(_query_unit, states=inputo)

			if self.drop is not None:
				context = self.drop(context)

			context = context + (_query_unit if self.norm_residual else query_unit)

		_context = self.layer_normer2(context)
		_context_new, _attn = self.cross_attn(_context, inpute, mask=src_pad_mask)

		if compute_ffn:
			if self.drop is not None:
				_context_new = self.drop(_context_new)

			context = _context_new + (_context if self.norm_residual else context)

			context = self.ff(context)
		else:
			context = _context_new

		if query_unit is None:
			return context, _attn
		else:
			return context, states_return, _attn

	def load_base(self, base_decoder_layer):

		self.self_attn = base_decoder_layer.self_attn
		self.cross_attn.load_base(base_decoder_layer.cross_attn)

		self.ff = base_decoder_layer.ff

		self.layer_normer1 = base_decoder_layer.layer_normer1
		self.layer_normer2 = base_decoder_layer.layer_normer2

		self.drop = base_decoder_layer.drop

		self.norm_residual = base_decoder_layer.norm_residual

class Decoder(DecoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, emb_w=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindemb=False, forbidden_index=None, **kwargs):

		_ahsize = isize if ahsize is None else ahsize

		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Decoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, emb_w=emb_w, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, bindemb=bindemb, forbidden_index=forbidden_index, **kwargs)

		self.nets = nn.ModuleList([DecoderLayer(isize, _fhsize, dropout, attn_drop, num_head, _ahsize) for i in range(num_layer)])

		self.tattn_w = nn.Parameter(torch.Tensor(num_layer * num_head).uniform_(- sqrt(1.0 / (num_layer * num_head)), sqrt(1.0 / (num_layer * num_head))))
		self.tattn_drop = Dropout(dropout) if dropout > 0.0 else None
		self.trans = Linear(isize, isize, bias=False)

	def forward(self, inpute, inputo, inputea, src_pad_mask=None):

		bsize, nquery = inputo.size()

		out = self.wemb(inputo)

		out = out * sqrt(out.size(-1))
		if self.pemb is not None:
			out = out + self.pemb(inputo, expand=False)

		if self.drop is not None:
			out = self.drop(out)

		_mask = self._get_subsequent_mask(nquery)

		attns = []
		_num_layer_s = len(self.nets) - 1
		for i, net in enumerate(self.nets):
			out, _attn = net(inpute, out, src_pad_mask, _mask, compute_ffn=i < _num_layer_s)
			attns.append(_attn)

		# attns: (bsize, num_layer * nheads, nquery, seql) => (bsize, nquery, seql, num_layer * nheads)
		attns = attns[0] if len(attns) == 1 else torch.cat(attns, dim=1)
		attns = attns.permute(0, 2, 3, 1).contiguous()
		_asize = attns.size()

		# inpute: (bsize, seql, isize)
		# attns: (bsize, nquery, seql, num_layer * nheads) => (bsize, nquery, seql)
		# out_enc: (bsize, nquery, isize)
		out_enc = attns.view(-1, _asize[-1]).mv(self.tattn_w.softmax(dim=0) if self.tattn_drop is None else self.tattn_drop(self.tattn_w).softmax(dim=0)).view(_asize[:-1]).bmm(inputea)

		if self.out_normer is not None:
			out_enc = self.out_normer(out_enc)

		return self.lsm(self.classifier(self.trans(out_enc)))

	def load_base(self, base_decoder):

		self.drop = base_decoder.drop

		self.wemb = base_decoder.wemb

		self.pemb = base_decoder.pemb

		for _net, _b_net in zip(self.nets, base_decoder.nets):
			_net.load_base(_b_net)

		self.classifier = base_decoder.classifier

		self.lsm = base_decoder.lsm

		self.out_normer = None if self.out_normer is None else base_decoder.out_normer
