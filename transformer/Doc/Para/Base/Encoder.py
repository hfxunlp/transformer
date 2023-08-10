#encoding: utf-8

import torch
from math import sqrt
from torch import nn

from modules.base import CrossAttn
from modules.paradoc import GateResidual
from transformer.Encoder import Encoder as EncoderBase, EncoderLayer as EncoderLayerBase
from utils.fmt.parser import parse_none
from utils.torch.comp import mask_tensor_type, torch_no_grad

from cnfg.ihyp import *

class CrossEncoderLayer(EncoderLayerBase):

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, ahsize=None, ncross=2, **kwargs):

		_ahsize = parse_none(ahsize, isize)

		super(CrossEncoderLayer, self).__init__(isize, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize)

		self.cattns = nn.ModuleList([CrossAttn(isize, _ahsize, isize, num_head, dropout=attn_drop) for i in range(ncross)])
		self.cattn_ln = nn.ModuleList([nn.LayerNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters) for i in range(ncross)])
		self.grs = nn.ModuleList([GateResidual(isize) for i in range(ncross)])

	def forward(self, inputs, inputc, mask=None, context_mask=None, **kwargs):

		_inputs = self.layer_normer(inputs)
		context = self.attn(_inputs, mask=mask)

		if self.drop is not None:
			context = self.drop(context)

		context = context + (_inputs if self.norm_residual else inputs)

		for _ln, _cattn, _gr, _inputc, _maskc in zip(self.cattn_ln, self.cattns, self.grs, inputc, [None for i in range(len(inputc))] if context_mask is None else context_mask):
			_inputs = _ln(context)
			_context = _cattn(_inputs, _inputc, mask=_maskc)
			if self.drop is not None:
				_context = self.drop(_context)
			context = _gr(_context, (_inputs if self.norm_residual else context))

		context = self.ff(context)

		return context

	def load_base(self, base_encoder_layer):

		self.drop = base_encoder_layer.drop
		self.attn = base_encoder_layer.attn
		self.ff = base_encoder_layer.ff
		self.layer_normer = base_encoder_layer.layer_normer

class CrossEncoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, nprev_context=2, **kwargs):

		_ahsize = parse_none(ahsize, isize)

		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(CrossEncoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output)

		self.nets = nn.ModuleList([CrossEncoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize) for i in range(num_layer)])

	def forward(self, inputs, inputc, mask=None, context_mask=None, **kwargs):

		out = self.wemb(inputs)
		if self.pemb is not None:
			out = self.pemb(inputs, expand=False).add(out, alpha=sqrt(out.size(-1)))

		if self.drop is not None:
			out = self.drop(out)

		for net in self.nets:
			out = net(out, inputc, mask, context_mask)

		return out if self.out_normer is None else self.out_normer(out)

	def load_base(self, base_encoder):

		self.drop = base_encoder.drop

		self.wemb = base_encoder.wemb

		self.pemb = base_encoder.pemb

		for snet, bnet in zip(self.nets, base_encoder.nets):
			snet.load_base(bnet)

		self.out_normer = None if self.out_normer is None else base_encoder.out_normer

class Encoder(nn.Module):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, nprev_context=2, num_layer_context=1, **kwargs):

		super(Encoder, self).__init__()

		_ahsize = parse_none(ahsize, isize)

		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		self.context_enc = EncoderBase(isize, nwd, num_layer if num_layer_context is None else num_layer_context, _fhsize, dropout, attn_drop, act_drop, num_head, xseql, _ahsize, norm_output)
		self.enc = CrossEncoder(isize, nwd, num_layer, _fhsize, dropout, attn_drop, act_drop, num_head, xseql, _ahsize, norm_output, nprev_context)

		_tmp_pad = torch.zeros(xseql, dtype=torch.long)
		_tmp_pad[0] = 1
		_tmp_pad = _tmp_pad.view(1, 1, xseql).repeat(1, nprev_context - 1, 1)
		self.register_buffer("pad", _tmp_pad, persistent=False)
		self.register_buffer("pad_mask", (1 - _tmp_pad).to(mask_tensor_type, non_blocking=True).unsqueeze(1), persistent=False)
		self.xseql = xseql

		self.nprev_context = nprev_context

	# inputs: (bsize, _nsent, seql), nprev_context, ... , nsent - 1
	# inputc: (bsize, _nsentc, seql), 0, 1, ... , nsent - 2
	# mask: (bsize, 1, _nsent, seql), generated with:
	#	mask = inputs.eq(pad_id).unsqueeze(1)
	# where _nsent = nsent - self.nprev_context, _nsentc = nsent - 1
	def forward(self, inputs, inputc, mask=None, context_mask=None, **kwargs):

		bsize, nsentc, seql = inputc.size()
		_inputc = torch.cat((self.get_pad(seql).expand(bsize, -1, seql), inputc,), dim=1)
		_context_mask = None if context_mask is None else torch.cat((self.get_padmask(seql).expand(bsize, 1, -1, seql), context_mask,), dim=2)
		context = self.context_enc(_inputc.view(-1, seql), mask=None if _context_mask is None else _context_mask.view(-1, 1, seql)).view(bsize, nsentc + self.nprev_context - 1, seql, -1)
		isize = context.size(-1)
		contexts = []
		context_masks = []
		for i in range(self.nprev_context):
			contexts.append(context.narrow(1, i, nsentc).contiguous().view(-1, seql, isize))
			context_masks.append(None if _context_mask is None else _context_mask.narrow(2, i, nsentc).contiguous().view(-1, 1, seql))

		seql = inputs.size(-1)

		return self.enc(inputs.view(-1, seql), contexts, None if mask is None else mask.view(-1, 1, seql), context_masks), contexts, context_masks

	def load_base(self, base_encoder):

		self.enc.load_base(base_encoder)
		with torch_no_grad():
			self.context_enc.wemb.weight.copy_(base_encoder.wemb.weight)

	def get_pad(self, seql):

		return self.pad.narrow(-1, 0, seql) if seql <= self.xseql else torch.cat((self.pad, self.pad.new_zeros(1, self.nprev_context - 1, seql - self.xseql),), dim=-1)

	def get_padmask(self, seql):

		return self.pad_mask.narrow(-1, 0, seql) if seql <= self.xseql else torch.cat((self.pad_mask, self.pad_mask.new_ones(1, 1, self.nprev_context - 1, seql - self.xseql),), dim=-1)

	def get_embedding_weight(self):

		return self.enc.get_embedding_weight()

	def update_vocab(self, indices):

		return self.enc.update_vocab(indices)
