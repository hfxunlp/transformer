#encoding: utf-8

import torch
#from math import sqrt
from torch import nn

from modules.act import Custom_Act, GEGLU, LGLU, get_act
from modules.base import CrossAttn as CrossAttnBase, Linear, PositionwiseFF as PositionwiseFFBase, ResCrossAttn as ResCrossAttnBase, ResSelfAttn as ResSelfAttnBase, SelfAttn as SelfAttnBase
from modules.dropout import Dropout
from modules.norm import RMSNorm as Norm
from utils.fmt.parser import parse_none
from utils.relpos.bucket import build_rel_pos_bucket, build_rel_pos_bucket_map

from cnfg.plm.t5.ihyp import *

class SelfAttn(SelfAttnBase):

	def __init__(self, isize, hsize, osize, num_head=8, dropout=0.0, enable_bias=enable_prev_ln_bias_default, enable_proj_bias=enable_proj_bias_default, k_rel_pos=use_k_relative_position, uni_direction_reduction=False, is_left_to_right_reduction=True, zero_reduction=relpos_reduction_with_zeros, max_bucket_distance=0, sparsenorm=False, xseql=cache_len_default, **kwargs):

		super(SelfAttn, self).__init__(isize, hsize, osize, num_head=num_head, dropout=dropout, enable_bias=enable_bias, enable_proj_bias=enable_proj_bias, k_rel_pos=k_rel_pos, uni_direction_reduction=uni_direction_reduction, is_left_to_right_reduction=is_left_to_right_reduction, zero_reduction=zero_reduction, max_bucket_distance=max_bucket_distance, sparsenorm=sparsenorm, xseql=xseql, **kwargs)

		self.ref_rel_emb = None
		self.rel_emb_cache = None

	def forward(self, iQ, mask=None, states=None, **kwargs):

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

		if self.ref_rel_emb is not None:
			scores += self.ref_rel_emb.rel_emb_cache
		elif self.rel_pemb is not None:
			if states is None:
				self.rel_pos_cache = self.get_rel_pos(nquery).contiguous() if self.ref_rel_posm is None else self.ref_rel_posm.rel_pos_cache
				self.rel_emb_cache = (real_iQ.permute(2, 0, 1, 3).contiguous().view(nquery, bsize * nheads, adim).bmm(self.rel_pemb(self.rel_pos_cache).transpose(1, 2)).view(nquery, bsize, nheads, nquery).permute(1, 2, 0, 3) if self.rel_pos_map is None else self.rel_pemb(self.rel_pos_cache).permute(2, 0, 1)).contiguous()
			else:
				self.rel_pos_cache = self.get_rel_pos(seql).narrow(0, seql - nquery, nquery).contiguous() if self.ref_rel_posm is None else self.ref_rel_posm.rel_pos_cache
				self.rel_emb_cache = (real_iQ.permute(2, 0, 1, 3).contiguous().view(nquery, bsize * nheads, adim).bmm(self.rel_pemb(self.rel_pos_cache).transpose(1, 2)).view(nquery, bsize, nheads, seql).permute(1, 2, 0, 3) if self.rel_pos_map is None else self.rel_pemb(self.rel_pos_cache).permute(2, 0, 1)).contiguous()
			scores += self.rel_emb_cache

		## t5 does not scale attention scores
		#scores = scores / sqrt(adim)

		if mask is not None:
			scores.masked_fill_(mask.unsqueeze(1), -inf_default)

		scores = self.normer(scores)

		if self.drop is not None:
			scores = self.drop(scores)

		out = self.outer(scores.matmul(real_iV).transpose(1, 2).contiguous().view(bsize, nquery, self.hsize))

		if states is None:
			return out
		else:
			return out, (real_iK, real_iV,)

class CrossAttn(CrossAttnBase):

	def __init__(self, isize, hsize, osize, num_head=8, dropout=0.0, k_isize=None, enable_bias=enable_prev_ln_bias_default, enable_proj_bias=enable_proj_bias_default, sparsenorm=False, k_rel_pos=use_k_relative_position, max_bucket_distance=0, xseql=cache_len_default, **kwargs):

		super(CrossAttn, self).__init__(isize, hsize, osize, num_head=num_head, dropout=dropout, k_isize=k_isize, enable_bias=enable_bias, enable_proj_bias=enable_proj_bias, sparsenorm=sparsenorm, **kwargs)

		if (k_rel_pos > 0) and (max_bucket_distance > 0):
			self.rel_shift = k_rel_pos
			self.register_buffer("rel_pos_map", build_rel_pos_bucket_map(k_rel_pos=k_rel_pos, max_len=max_bucket_distance, uni_direction=False), persistent=False)
			self.register_buffer("rel_pos", build_rel_pos_bucket(xseql, k_rel_pos=k_rel_pos, max_len=max_bucket_distance, uni_direction=False, dis_map=self.rel_pos_map), persistent=False)
			self.rel_pemb = nn.Embedding(k_rel_pos + k_rel_pos + 1, self.num_head)
			self.clamp_max, self.clamp_min = max_bucket_distance, False
			self.xseql = xseql
			self.ref_rel_posm = None
			self.register_buffer("rel_pos_cache", None, persistent=False)
		else:
			self.rel_pemb = None
		self.ref_rel_emb = None
		self.rel_emb_cache = None

	def forward(self, iQ, iK, mask=None, step=0, **kwargs):

		bsize, nquery = iQ.size()[:2]
		seql = iK.size(1)
		nheads = self.num_head
		adim = self.attn_dim

		real_iQ = self.query_adaptor(iQ).view(bsize, nquery, nheads, adim).transpose(1, 2)
		if (self.real_iK is not None) and self.iK.is_set_to(iK) and self.is_decoding:
			real_iK, real_iV = self.real_iK, self.real_iV
		else:
			real_iK, real_iV = self.kv_adaptor(iK).view(bsize, seql, 2, nheads, adim).unbind(2)
			real_iK, real_iV = real_iK.permute(0, 2, 3, 1), real_iV.transpose(1, 2)
			if self.is_decoding:
				self.iK, self.real_iK, self.real_iV = iK, real_iK, real_iV

		scores = real_iQ.matmul(real_iK)

		if self.ref_rel_emb is not None:
			scores += self.ref_rel_emb.rel_emb_cache
		elif self.rel_pemb is not None:
			self.rel_pos_cache = (self.get_rel_pos(step, seql).narrow(0, step - nquery, nquery) if step > 0 else self.get_rel_pos(nquery, seql)).contiguous() if self.ref_rel_posm is None else self.ref_rel_posm.rel_pos_cache
			self.rel_emb_cache = self.rel_pemb(self.rel_pos_cache).permute(2, 0, 1).contiguous()
			scores += self.rel_emb_cache

		# t5 does not scale attention scores
		#scores = scores / sqrt(adim)

		if mask is not None:
			scores.masked_fill_(mask.unsqueeze(1), -inf_default)

		scores = self.normer(scores)

		if self.drop is not None:
			scores = self.drop(scores)

		return self.outer(scores.matmul(real_iV).transpose(1, 2).contiguous().view(bsize, nquery, self.hsize))

	def get_rel_pos(self, length, seql):

		_ = max(length, seql)
		if _ <= self.xseql:
			return self.rel_pos.narrow(0, 0, length).narrow(1, 0, seql)
		else:
			_out = build_rel_pos_bucket(_, k_rel_pos=self.rel_shift, max_len=self.clamp_max, uni_direction=self.clamp_min, device=self.rel_pos.device, dis_map=self.rel_pos_map)
			if length < _:
				_out = _out.narrow(0, 0, length)
			elif seql < _:
				_out = _out.narrow(1, 0, seql)
			return _out

class ResSelfAttn(ResSelfAttnBase):

	def __init__(self, isize, hsize, num_head=8, dropout=0.0, norm_residual=norm_residual_default, **kwargs):

		super(ResSelfAttn, self).__init__(isize, hsize, num_head=num_head, dropout=dropout, norm_residual=norm_residual, **kwargs)

		self.net = SelfAttn(isize, hsize, isize, num_head=num_head, dropout=dropout, **kwargs)
		self.normer = Norm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)

class ResCrossAttn(ResCrossAttnBase):

	def __init__(self, isize, hsize, num_head=8, dropout=0.0, norm_residual=norm_residual_default, **kwargs):

		super(ResCrossAttn, self).__init__(isize, hsize, num_head=num_head, dropout=dropout, norm_residual=norm_residual, **kwargs)

		self.net = CrossAttn(isize, hsize, isize, num_head=num_head, dropout=dropout, **kwargs)
		self.normer = Norm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)

class PositionwiseFF(PositionwiseFFBase):

	def __init__(self, isize, hsize=None, dropout=0.0, act_drop=None, norm_residual=norm_residual_default, custom_act=use_adv_act_default, enable_bias=enable_prev_ln_bias_default, use_glu=use_glu_ffn, **kwargs):

		_hsize = isize * 4 if hsize is None else hsize
		_act_drop = parse_none(act_drop, dropout)

		super(PositionwiseFF, self).__init__(isize, hsize=_hsize, dropout=dropout, act_drop=_act_drop, norm_residual=norm_residual, custom_act=custom_act, enable_bias=enable_bias, use_glu=None, **kwargs)

		if (use_glu is not None) and (_hsize % 2 == 1):
			_hsize += 1

		_ = [Linear(isize, _hsize, bias=enable_bias)]
		_drop_ind = 2
		if use_glu is None:
			_.extend([Custom_Act() if custom_act else nn.ReLU(inplace=True), Linear(_hsize, isize, bias=enable_bias)])
		else:
			use_glu = use_glu.lower()
			if use_glu == "glu":
				_.append(nn.GLU())
			elif use_glu == "geglu":
				_.append(GEGLU())
			else:
				_act = get_act(use_glu, None)
				if _act is not None:
					_.append(_act())
					_drop_ind += 1
				_.append(LGLU())
			_.append(Linear(_hsize // 2, isize, bias=enable_bias))
		if dropout > 0.0:
			_.append(Dropout(dropout, inplace=True))
		if _act_drop > 0.0:
			_.insert(_drop_ind, Dropout(_act_drop, inplace=inplace_after_Custom_Act))
		self.net = nn.Sequential(*_)
		self.normer = Norm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)
