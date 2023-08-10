#encoding: utf-8

import torch
from math import ceil, sqrt
from torch import nn

from modules.base import CrossAttn, Dropout, ResidueCombiner
from transformer.TA.Encoder import Encoder as EncoderBase, EncoderLayer as EncoderLayerBase
from utils.fmt.parser import parse_none

from cnfg.ihyp import *

class EncoderLayer(EncoderLayerBase):

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, ahsize=None, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(EncoderLayer, self).__init__(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize, **kwargs)

		self.cattn = CrossAttn(isize, _ahsize, isize, num_head, dropout=attn_drop)
		self.scff = ResidueCombiner(isize, 2, _fhsize, dropout)

	def forward(self, inputs, sumr, mask=None, rmask=None, **kwargs):

		#_bsize, _seql, _isize = inputs.size()
		#_rep1, _rep2 = self.cattn(inputs.repeat(2, 1, 1), sumr, rmask).view(2, _bsize, _seql, _isize).unbind(0)
		#inputs = self.scff(inputs, _rep1, _rep2)
		inputs = self.scff(inputs, self.cattn(inputs, sumr, rmask))

		context = self.attn(inputs, mask=mask)

		context = self.ff(context)

		return context

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, share_layer=False, num_layer_dec=6, max_chunk_tokens=8, min_chunks=4, **kwargs):

		_ahsize = parse_none(ahsize, isize)

		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Encoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, share_layer=share_layer, num_layer_dec=num_layer_dec, **kwargs)

		if share_layer:
			_shared_layer = EncoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize)
			self.nets = nn.ModuleList([_shared_layer for i in range(num_layer)])
		else:
			self.nets = nn.ModuleList([EncoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize) for i in range(num_layer)])

		self.sc_tattn_w = nn.Parameter(torch.Tensor(num_layer + 1, num_layer_dec).uniform_(- sqrt(1.0 / (num_layer + 1)), sqrt(1.0 / (num_layer + 1))))
		self.sc_tattn_drop = Dropout(dropout) if dropout > 0.0 else None

		self.mxct = max_chunk_tokens
		self.mnck = float(min_chunks)

	def forward(self, inputs, mask=None, **kwargs):

		def transform(lin, w, drop):

			_tmp = torch.stack(lin, -1)
			_osize = _tmp.size()
			_tmp = _tmp.view(-1, _osize[-1]).mm(w.softmax(dim=0) if drop is None else drop(w).softmax(dim=0))
			_osize = list(_osize)
			_osize[-1] = -1

			return _tmp.view(_osize)

		def build_chunk_max(rept, bsize, nchk, ntok, npad, mask=None, rmask=None):

			out = rept.masked_fill(mask.squeeze(1).unsqueeze(-1), -inf_default) if npad == 0 else torch.cat((rept.masked_fill(mask.squeeze(1).unsqueeze(-1), -inf_default), rept.new_full((bsize, npad, rept.size(-1)), -inf_default),), dim=1)

			return out.view(bsize, nchk, ntok, -1).max(2)[0].masked_fill(rmask.squeeze(1).unsqueeze(-1), 0.0)

		def build_chunk_mean(rept, bsize, nchk, ntok, npad, mask=None, rmask=None, nele=None):

			out = rept.masked_fill(mask.squeeze(1).unsqueeze(-1), 0.0) if npad == 0 else torch.cat((rept.masked_fill(mask.squeeze(1).unsqueeze(-1), 0.0), rept.new_zeros((bsize, npad, rept.size(-1))),), dim=1)

			return (out.view(bsize, nchk, ntok, -1).sum(2) / nele).masked_fill(rmask.squeeze(1).unsqueeze(-1), 0.0)

		bsize, seql = inputs.size()

		_ntok = max(min(self.mxct, ceil(seql / self.mnck)), 2)
		_npad = (_ntok - (seql % _ntok)) % _ntok
		_nchk = int((seql + _npad) / _ntok)
		if mask is None:
			_chk_mask = None
			_rmask = None
		else:
			_chk_mask = mask if _npad == 0 else torch.cat((mask, mask.new_ones(bsize, 1, _npad),), dim=-1)
			_nmask = _chk_mask.view(bsize, 1, _nchk, _ntok).sum(-1)
			_rmask = _nmask.ge(_ntok)

		out = self.wemb(inputs)
		if self.pemb is not None:
			out = self.pemb(inputs, expand=False).add(out, alpha=sqrt(out.size(-1)))

		#if _rmask is not None:
			#_nele = (_ntok - _nmask).view(bsize, _nchk, 1).to(out, non_blocking=True)

		if self.drop is not None:
			out = self.drop(out)

		out = self.out_normer(out)
		outs = [out]

		_ho = build_chunk_max(out, bsize, _nchk, _ntok, _npad, mask, _rmask)
		#_ho = torch.cat((build_chunk_mean(out, bsize, _nchk, _ntok, _npad, mask, _srmask, _nele), build_chunk_max(out, bsize, _nchk, _ntok, _npad, mask, _srmask),), 0)
		hl = [_ho]

		for net in self.nets:
			out = net(out, _ho, mask, _rmask)
			outs.append(out)
			_ho = build_chunk_max(out, bsize, _nchk, _ntok, _npad, mask, _rmask)
			#_ho = torch.cat((build_chunk_mean(out, bsize, _nchk, _ntok, _npad, mask, _srmask, _nele), build_chunk_max(out, bsize, _nchk, _ntok, _npad, mask, _srmask),), 0)
			hl.append(_ho)

		out = transform(outs, self.tattn_w, self.tattn_drop)

		# hl: (bsize, _nchk, isize, num_layer + 1)
		hl = transform(hl, self.sc_tattn_w, self.sc_tattn_drop)

		return out, hl, _rmask
