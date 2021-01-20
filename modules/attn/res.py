#encoding: utf-8

from torch.nn import functional as nnFunc
from math import sqrt

from modules.base import SelfAttn as SelfAttnBase
from modules.base import CrossAttn as CrossAttnBase

from cnfg.ihyp import *

class SelfAttn(SelfAttnBase):

	def forward(self, iQ, mask=None, iK=None, resin=None):

		bsize, nquery = iQ.size()[:2]
		nheads = self.num_head
		adim = self.attn_dim

		if iK is None:

			real_iQ, real_iK, real_iV = self.adaptor(iQ).view(bsize, nquery, 3, nheads, adim).unbind(2)

		else:

			seql = iK.size(1)

			real_iQ, _out = nnFunc.linear(iQ, self.adaptor.weight.narrow(0, 0, self.hsize), None if self.adaptor.bias is None else self.adaptor.bias.narrow(0, 0, self.hsize)).view(bsize, nquery, nheads, adim), nnFunc.linear(iK, self.adaptor.weight.narrow(0, self.hsize, self.hsize + self.hsize), None if self.adaptor.bias is None else self.adaptor.bias.narrow(0, self.hsize, self.hsize + self.hsize)).view(bsize, seql, 2, nheads, adim)
			real_iK, real_iV = _out.unbind(2)

		real_iQ, real_iK, real_iV = real_iQ.transpose(1, 2), real_iK.permute(0, 2, 3, 1), real_iV.transpose(1, 2)

		scores = real_iQ.matmul(real_iK)

		if self.rel_pemb is not None:
			if iK is None:
				self.rel_pos_cache = self.get_rel_pos(nquery).contiguous() if self.ref_rel_posm is None else self.ref_rel_posm.rel_pos_cache
				scores += real_iQ.permute(2, 0, 1, 3).contiguous().view(nquery, bsize * nheads, adim).bmm(self.rel_pemb(self.rel_pos_cache).transpose(1, 2)).view(nquery, bsize, nheads, nquery).permute(1, 2, 0, 3)
			else:
				self.rel_pos_cache = self.get_rel_pos(seql).narrow(0, seql - nquery, nquery).contiguous() if self.ref_rel_posm is None else self.ref_rel_posm.rel_pos_cache
				scores += real_iQ.permute(2, 0, 1, 3).contiguous().view(nquery, bsize * nheads, adim).bmm(self.rel_pemb(self.rel_pos_cache).transpose(1, 2)).view(nquery, bsize, nheads, seql).permute(1, 2, 0, 3)

		scores = scores / sqrt(adim)

		if mask is not None:
			scores.masked_fill_(mask.unsqueeze(1), -inf_default)

		if resin is None:
			resout = scores
		else:
			resout = scores = scores + resin

		scores = self.normer(scores)

		if self.drop is not None:
			scores = self.drop(scores)

		oMA = scores.matmul(real_iV).transpose(1, 2).contiguous()

		return self.outer(oMA.view(bsize, nquery, self.hsize)), resout

class CrossAttn(CrossAttnBase):

	def forward(self, iQ, iK, mask=None, resin=None):

		bsize, nquery = iQ.size()[:2]
		seql = iK.size(1)
		nheads = self.num_head
		adim = self.attn_dim

		real_iQ, _out = self.query_adaptor(iQ).view(bsize, nquery, nheads, adim), self.kv_adaptor(iK).view(bsize, seql, 2, nheads, adim)
		real_iK, real_iV = _out.unbind(2)

		real_iQ, real_iK, real_iV = real_iQ.transpose(1, 2), real_iK.permute(0, 2, 3, 1), real_iV.transpose(1, 2)

		scores = real_iQ.matmul(real_iK) / sqrt(adim)

		if mask is not None:
			scores.masked_fill_(mask.unsqueeze(1), -inf_default)

		if resin is None:
			resout = scores
		else:
			resout = scores = scores + resin

		scores = self.normer(scores)

		if self.drop is not None:
			scores = self.drop(scores)

		oMA = scores.matmul(real_iV).transpose(1, 2).contiguous()

		return self.outer(oMA.view(bsize, nquery, self.hsize)), resout
