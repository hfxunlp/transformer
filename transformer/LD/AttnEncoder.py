#encoding: utf-8

import torch
from torch import nn
from modules.LD import ATTNCombiner

from math import sqrt, ceil

from transformer.LD.Encoder import Encoder as EncoderBase

from cnfg.ihyp import *

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, *inputs, **kwargs):

		super(Encoder, self).__init__(isize, nwd, num_layer, *inputs, **kwargs)

		self.attn_emb = ATTNCombiner(isize, isize, dropout)
		self.attns = nn.ModuleList([ATTNCombiner(isize, isize, dropout) for i in range(num_layer)])

	def forward(self, inputs, mask=None):

		def transform(lin, w, drop):

			_tmp = torch.stack(lin, -1)
			_osize = _tmp.size()
			_tmp = _tmp.view(-1, _osize[-1]).mm(w.softmax(dim=0) if drop is None else drop(w).softmax(dim=0))
			_osize = list(_osize)
			_osize[-1] = -1

			return _tmp.view(_osize)

		def build_chunk_max(atm, rept, bsize, nchk, ntok, npad, mask=None, rmask=None, chkmask=None):

			pad_out = rept.masked_fill(mask.squeeze(1).unsqueeze(-1), -inf_default) if npad == 0 else torch.cat((rept.masked_fill(mask.squeeze(1).unsqueeze(-1), -inf_default), rept.new_full((bsize, npad, rept.size(-1)), -inf_default),), dim=1)

			# query: bsize, nchk, isize
			# kv: bsize, nchk*ntok, isize
			query = pad_out.view(bsize, nchk, ntok, -1).max(2)[0].masked_fill(rmask.view(bsize, -1, 1), 0.0)
			kv = rept.masked_fill(mask.squeeze(1).unsqueeze(-1), 0.0) if npad == 0 else torch.cat((rept.masked_fill(mask.squeeze(1).unsqueeze(-1), 0.0), rept.new_zeros((bsize, npad, rept.size(-1))),), dim=1)
			out = atm(query.view(bsize * nchk, 1, -1), kv.view(bsize * nchk, ntok, -1), chkmask.view(bsize * nchk, ntok, 1)).view(bsize, nchk, -1)

			# mask is not necessary in theory .masked_fill(rmask.squeeze(1).unsqueeze(-1), 0.0)
			return out

		def build_chunk_mean(atm, rept, bsize, nchk, ntok, npad, mask=None, rmask=None, chkmask=None, nele=None):

			pad_out = rept.masked_fill(mask.squeeze(1).unsqueeze(-1), 0.0) if npad == 0 else torch.cat((rept.masked_fill(mask.squeeze(1).unsqueeze(-1), 0.0), rept.new_zeros((bsize, npad, rept.size(-1))),), dim=1)

			query = pad_out.view(bsize, nchk, ntok, -1).sum(2) / nele
			out = atm(query.view(bsize * nchk, 1, -1), pad_out.view(bsize * nchk, ntok, -1), chkmask.view(bsize * nchk, ntok, 1)).view(bsize, nchk, -1)

			return out

		bsize, seql = inputs.size()

		_ntok = max(min(self.mxct, ceil(seql / self.mnck)), 3)
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
		out = out * sqrt(out.size(-1))
		if self.pemb is not None:
			out = out + self.pemb(inputs, expand=False)

		#if _rmask is not None:
			#_nele = (_ntok - _nmask).masked_fill(_nmask.eq(_ntok), 1).view(bsize, _nchk, 1).to(out)

		if self.drop is not None:
			out = self.drop(out)

		out = self.out_normer(out)
		outs = [out]

		_ho = build_chunk_max(self.attn_emb, out, bsize, _nchk, _ntok, _npad, mask, _rmask, _chk_mask)
		#_ho = build_chunk_mean(self.attn_emb, out, bsize, _nchk, _ntok, _npad, mask, _rmask, _chk_mask, _nele)
		hl = [_ho]

		for net, attnm in zip(self.nets, self.attns):
			out = net(out, _ho, mask, _rmask)
			outs.append(out)
			_ho = build_chunk_max(attnm, out, bsize, _nchk, _ntok, _npad, mask, _rmask, _chk_mask)
			#_ho = build_chunk_mean(attnm, out, bsize, _nchk, _ntok, _npad, mask, _rmask, _chk_mask, _nele)
			hl.append(_ho)

		out = transform(outs, self.tattn_w, self.tattn_drop)

		# hl: (bsize, _nchk, isize, num_layer + 1)
		hl = transform(hl, self.sc_tattn_w, self.sc_tattn_drop)

		return out, hl, _rmask
