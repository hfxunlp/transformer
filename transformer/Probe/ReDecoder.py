#encoding: utf-8

import torch
from torch import nn

from math import sqrt

from modules.base import Linear

from transformer.Decoder import DecoderLayer as DecoderLayerBase
from transformer.Decoder import Decoder as DecoderBase

from cnfg.ihyp import *

class DecoderLayer(DecoderLayerBase):

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, num_head=8, ahsize=None, **kwargs):

		super(DecoderLayer, self).__init__(isize, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, num_head=num_head, ahsize=ahsize, **kwargs)

		self.perform_self_attn = True
		self.perform_cross_attn = True

	def forward(self, inpute, inputo, src_pad_mask=None, tgt_pad_mask=None, query_unit=None):

		if query_unit is None:
			if self.perform_self_attn:
				_inputo = self.layer_normer1(inputo)

				context = self.self_attn(_inputo, mask=tgt_pad_mask)
				if self.drop is not None:
					context = self.drop(context)
				context = context + (_inputo if self.norm_residual else inputo)
			else:
				context, states_return = inputo, None

		else:
			if self.perform_self_attn:
				_query_unit = self.layer_normer1(query_unit)
				context, states_return = self.self_attn(_query_unit, states=inputo)
				if self.drop is not None:
					context = self.drop(context)
				context = context + (_query_unit if self.norm_residual else query_unit)
			else:
				context, states_return = query_unit, query_unit if inputo is None else torch.cat((inputo, query_unit,), 1)

		if self.perform_cross_attn:
			_context = self.layer_normer2(context)
			_context_new = self.cross_attn(_context, inpute, mask=src_pad_mask)
			if self.drop is not None:
				_context_new = self.drop(_context_new)
			context = _context_new + (_context if self.norm_residual else context)

		context = self.ff(context)

		if query_unit is None:
			return context
		else:
			return context, states_return

	def load_base(self, base_decoder_layer):

		self.self_attn = base_decoder_layer.self_attn
		self.cross_attn = base_decoder_layer.cross_attn
		self.ff = base_decoder_layer.ff
		self.layer_normer1 = base_decoder_layer.layer_normer1
		self.layer_normer2 = base_decoder_layer.layer_normer2
		self.drop = base_decoder_layer.drop
		self.norm_residual = base_decoder_layer.norm_residual

class Decoder(DecoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, emb_w=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindemb=False, forbidden_index=None, num_layer_ana=None, **kwargs):

		_ahsize = isize if ahsize is None else ahsize
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Decoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, emb_w=emb_w, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, bindemb=bindemb, forbidden_index=forbidden_index, **kwargs)
		self.nets = nn.ModuleList([DecoderLayer(isize, _fhsize, dropout, attn_drop, num_head, _ahsize) for i in range(num_layer)])

		self.trans = Linear(isize, isize, bias=False)
		_num_layer_ana = num_layer if num_layer_ana is None else num_layer_ana
		self.nets = nn.ModuleList(list(self.nets[:_num_layer_ana])) if _num_layer_ana > 0 else None

	def forward(self, inpute, inputo, src_pad_mask=None):

		nquery = inputo.size(-1)

		out = self.wemb(inputo)

		out = out * sqrt(out.size(-1))
		if self.pemb is not None:
			out = out + self.pemb(inputo, expand=False)

		if self.drop is not None:
			out = self.drop(out)

		if self.nets is not None:
			_mask = self._get_subsequent_mask(nquery)
			for net in self.nets:
				out = net(inpute, out, src_pad_mask, _mask)

		if self.out_normer is not None:
			out = self.out_normer(out)

		out = self.lsm(self.classifier(self.trans(out)))

		return out

	def load_base(self, base_decoder):

		self.drop = base_decoder.drop

		self.wemb = base_decoder.wemb

		self.pemb = base_decoder.pemb

		_nets = list(base_decoder.nets)

		if self.nets is not None:
			for _net, _b_net in zip(self.nets, base_decoder.nets):
				_net.load_base(_b_net)

		self.classifier = base_decoder.classifier

		self.lsm = base_decoder.lsm

		self.out_normer = None if self.out_normer is None else base_decoder.out_normer
