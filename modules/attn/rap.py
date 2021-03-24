#encoding: utf-8

from math import sqrt

import torch
from torch import nn
from torch.autograd import Function

from modules.base import CrossAttn as CrossAttnBase
from modules.base import SelfAttn as SelfAttnBase

from cnfg.ihyp import *

class SelfAttn(SelfAttnBase):

	def forward(self, iQ, mask=None, states=None):

		bsize, nquery = iQ.size()[:2]
		nheads = self.num_head
		adim = self.attn_dim

		real_iQ, real_iK, real_iV = self.adaptor(iQ).view(bsize, nquery, 3, nheads, adim).unbind(2)
		real_iQ, real_iK, real_iV = real_iQ.transpose(1, 2), real_iK.permute(0, 2, 3, 1), real_iV.transpose(1, 2)

		if states is not None:
			_h_real_iK, _h_real_iV = states
			if _h_real_iK is None:
				seql = nquery
			else:
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

		_rscore = scores = scores / sqrt(adim)

		if mask is not None:
			scores.masked_fill_(mask.unsqueeze(1), -inf_default)

		scores = self.normer(scores)

		if self.drop is not None:
			scores = self.drop(scores)

		out = self.outer(scores.matmul(real_iV).transpose(1, 2).contiguous().view(bsize, nquery, self.hsize))

		if states is None:
			return out, _rscore
		else:
			return out, (real_iK, real_iV,), _rscore

	def load_base(self, base_module):

		self.attn_dim, self.hsize, self.num_head = base_module.attn_dim, base_module.hsize, base_module.num_head

		self.adaptor = base_module.adaptor

		self.outer = base_module.outer

		self.normer = base_module.normer

		self.drop = base_module.drop

		self.rel_pemb = base_module.rel_pemb
		if self.rel_pemb is not None:
			self.k_rel_pos, self.xseql = base_module.k_rel_pos, base_module.xseql
			self.ref_rel_posm = base_module.ref_rel_posm
			self.register_buffer("rel_pos", base_module.rel_pos)
			self.register_buffer("rel_pos_cache", base_module.rel_pos_cache)

class CrossAttn(CrossAttnBase):

	def forward(self, iQ, iK, mask=None):

		bsize, nquery = iQ.size()[:2]
		seql = iK.size(1)
		nheads = self.num_head
		adim = self.attn_dim

		real_iQ = self.query_adaptor(iQ).view(bsize, nquery, nheads, adim).transpose(1, 2)
		if (self.real_iK is not None) and self.iK.is_set_to(iK) and (not self.training):
			real_iK, real_iV = self.real_iK, self.real_iV
		else:
			real_iK, real_iV = self.kv_adaptor(iK).view(bsize, seql, 2, nheads, adim).unbind(2)
			real_iK, real_iV = real_iK.permute(0, 2, 3, 1), real_iV.transpose(1, 2)
			if not self.training:
				self.iK, self.real_iK, self.real_iV = iK, real_iK, real_iV

		scores = real_iQ.matmul(real_iK) / sqrt(adim)

		if mask is not None:
			scores.masked_fill_(mask.unsqueeze(1), -inf_default)

		_rscore = scores = self.normer(scores)

		if self.drop is not None:
			scores = self.drop(scores)

		return self.outer(scores.matmul(real_iV).transpose(1, 2).contiguous().view(bsize, nquery, self.hsize)), _rscore

	def load_base(self, base_module):

		self.attn_dim, self.hsize, self.num_head = base_module.attn_dim, base_module.hsize, base_module.num_head

		self.query_adaptor = base_module.query_adaptor

		self.kv_adaptor = base_module.kv_adaptor

		self.outer = base_module.outer

		self.normer = base_module.normer

		self.drop = base_module.drop
