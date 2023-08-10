#encoding: utf-8

import torch
from math import sqrt

from modules.base import CrossAttn as CrossAttnBase, ResCrossAttn as ResCrossAttnBase, ResSelfAttn as ResSelfAttnBase, SelfAttn as SelfAttnBase
from modules.sampler import Retriever
from utils.torch.comp import exist_any, torch_no_grad

from cnfg.ihyp import *

class SelfAttn(SelfAttnBase):

	def __init__(self, *args, xseql=cache_len_default, smoothing=None, use_cumsum=False, **kwargs):

		super(SelfAttn, self).__init__(*args, **kwargs)

		self.retriever = Retriever()
		self.smoothing, self.use_cumsum = smoothing if (smoothing is not None) and (smoothing > 0.0) and (smoothing < 1.0) else None, use_cumsum
		if self.use_cumsum and (self.smoothing is not None):
			self.num_pos = xseql
			self.register_buffer("csum", torch.Tensor(xseql, 1), persistent=False)
			self.reset_parameters()
		else:
			self.register_buffer("csum", None, persistent=False)

	def forward(self, iQ, mask=None, states=None, **kwargs):

		bsize, nquery = iQ.size()[:2]
		seql = nquery
		nheads = self.num_head
		adim = self.attn_dim

		real_iQ, real_iK, real_iV = self.adaptor(iQ).view(bsize, nquery, 3, nheads, adim).unbind(2)

		real_iQ, real_iK, real_iV = real_iQ.transpose(1, 2), real_iK.permute(0, 2, 3, 1), real_iV.transpose(1, 2).contiguous()
		if states is not None:
			_h_real_iK, _h_real_iV = states
			if _h_real_iK is not None:
				seql = nquery + _h_real_iK.size(-1)
				real_iK, real_iV = torch.cat((_h_real_iK, real_iK,), dim=-1), torch.cat((_h_real_iV, real_iV,), dim=2)

		scores = real_iQ.matmul(real_iK)

		if self.rel_pemb is not None:
			if states is None:
				self.rel_pos_cache = self.get_rel_pos(nquery).contiguous() if self.ref_rel_posm is None else self.ref_rel_posm.rel_pos_cache
				scores += real_iQ.permute(2, 0, 1, 3).contiguous().view(nquery, bsize * nheads, adim).bmm(self.rel_pemb(self.rel_pos_cache).transpose(1, 2)).view(nquery, bsize, nheads, nquery).permute(1, 2, 0, 3)
			else:
				self.rel_pos_cache = self.get_rel_pos(seql).narrow(0, seql - nquery, nquery).contiguous() if self.ref_rel_posm is None else self.ref_rel_posm.rel_pos_cache
				scores += real_iQ.permute(2, 0, 1, 3).contiguous().view(nquery, bsize * nheads, adim).bmm(self.rel_pemb(self.rel_pos_cache).transpose(1, 2)).view(nquery, bsize, nheads, seql).permute(1, 2, 0, 3)

		if self.training:
			scores = scores / sqrt(adim)

			if mask is not None:
				_mask = mask.unsqueeze(1)
				scores.masked_fill_(_mask, -inf_default)

			scores = self.normer(scores)

			if self.drop is not None:
				scores = self.drop(scores)
				# prevent all-zero dropout breaking multinomial sampling.
				scores.select(-1, 0).add_(ieps_dropout_multinomial_default)
				"""if mask is None:
					scores.add_(ieps_dropout_multinomial_default)
				else:
					scores[(~_mask).expand(scores.size())] += ieps_dropout_multinomial_default"""
		else:
			if mask is not None:
				scores.masked_fill_(mask.unsqueeze(1), -inf_default)

		oMA = self.retriever(scores, real_iV)

		if self.smoothing is not None:
			if self.use_cumsum:
				if states is None:
					_len = self.get_ext(seql) if seql > self.num_pos else self.csum.narrow(0, 0, seql)
					real_iV = real_iV.cumsum(2)
				else:
					_len = float(seql - 1) if seql > 1 else 1.0
					real_iV = real_iV.sum(2, keepdim=True)
			else:
				if mask is None:
					_len = float(seql - 1) if seql > 1 else 1.0
				else:
					_mask = mask.view(bsize, 1, seql, 1)
					_len = float(seql - 1) - _mask.to(oMA.dtype, non_blocking=True).sum(2, keepdim=True)
					_len_zm = _len.eq(0.0)
					if exist_any(_len_zm):
						_len.masked_fill_(_len_zm, 1.0)
					real_iV = real_iV.masked_fill(_mask, 0.0)
				real_iV = real_iV.sum(2, keepdim=True)
			_smoothing_w = self.smoothing / _len
			oMA = oMA * (1.0 - self.smoothing - _smoothing_w) + real_iV.mul_(_smoothing_w)

		oMA = oMA.transpose(1, 2).contiguous()

		out = self.outer(oMA.view(bsize, nquery, self.hsize))

		if states is None:
			return out
		else:
			return out, (real_iK, real_iV,)

	def reset_parameters(self):

		if self.csum is not None:
			self.csum = self.get_ext(self.num_pos)

	def get_ext(self, npos):

		_rs = torch.arange(npos, dtype=self.csum.dtype, device=self.csum.device)
		with torch_no_grad():
			_rs[0] = 1.0

		return _rs.unsqueeze(-1)

class CrossAttn(CrossAttnBase):

	def __init__(self, *args, smoothing=None, **kwargs):

		super(CrossAttn, self).__init__(*args, **kwargs)

		self.retriever = Retriever()
		self.smoothing = smoothing if (smoothing is not None) and (smoothing > 0.0) and (smoothing < 1.0) else None

	def forward(self, iQ, iK, mask=None, **kwargs):

		bsize, nquery = iQ.size()[:2]
		seql = iK.size(1)
		nheads = self.num_head
		adim = self.attn_dim

		real_iQ = self.query_adaptor(iQ).view(bsize, nquery, nheads, adim).transpose(1, 2)
		if (self.real_iK is not None) and self.iK.is_set_to(iK) and self.is_decoding:
			real_iK, real_iV = self.real_iK, self.real_iV
		else:
			real_iK, real_iV = self.kv_adaptor(iK).view(bsize, seql, 2, nheads, adim).unbind(2)
			real_iK, real_iV = real_iK.permute(0, 2, 3, 1), real_iV.transpose(1, 2).contiguous()
			if self.is_decoding:
				self.iK, self.real_iK, self.real_iV = iK, real_iK, real_iV

		scores = real_iQ.matmul(real_iK)

		if self.training:
			scores = scores / sqrt(adim)

			if mask is not None:
				_mask = mask.unsqueeze(1)
				scores.masked_fill_(_mask, -inf_default)

			scores = self.normer(scores)

			if self.drop is not None:
				scores = self.drop(scores)
				# prevent all-zero dropout breaking multinomial sampling.
				scores.select(-1, 0).add_(ieps_dropout_multinomial_default)
				"""if mask is None:
					scores.add_(ieps_dropout_multinomial_default)
				else:
					scores[(~_mask).expand(scores.size())] += ieps_dropout_multinomial_default"""
		else:
			if mask is not None:
				scores.masked_fill_(mask.unsqueeze(1), -inf_default)

		oMA = self.retriever(scores, real_iV)

		if self.smoothing is not None:
			if mask is None:
				_len = float(seql - 1) if seql > 1 else 1.0
			else:
				_mask = mask.view(bsize, 1, seql, 1)
				_len = float(seql - 1) - _mask.to(oMA.dtype, non_blocking=True).sum(2, keepdim=True)
				_len_zm = _len.eq(0.0)
				if exist_any(_len_zm):
					_len.masked_fill_(_len_zm, 1.0)
				real_iV = real_iV.masked_fill(_mask, 0.0)
			real_iV = real_iV.sum(2, keepdim=True)
			_smoothing_w = self.smoothing / _len
			oMA = oMA * (1.0 - self.smoothing - _smoothing_w) + real_iV.mul_(_smoothing_w)

		oMA = oMA.transpose(1, 2).contiguous()

		return self.outer(oMA.view(bsize, nquery, self.hsize))

class ResSelfAttn(ResSelfAttnBase):

	def __init__(self, isize, hsize, num_head=8, dropout=0.0, norm_residual=norm_residual_default, **kwargs):

		super(ResSelfAttn, self).__init__(isize, hsize, num_head=num_head, dropout=dropout, norm_residual=norm_residual, **kwargs)

		self.net = SelfAttn(isize, hsize, isize, num_head=num_head, dropout=dropout, **kwargs)

class ResCrossAttn(ResCrossAttnBase):

	def __init__(self, isize, hsize, num_head=8, dropout=0.0, norm_residual=norm_residual_default, **kwargs):

		super(ResCrossAttn, self).__init__(isize, hsize, num_head=num_head, dropout=dropout, norm_residual=norm_residual, **kwargs)

		self.net = CrossAttn(isize, hsize, isize, num_head=num_head, dropout=dropout, **kwargs)
