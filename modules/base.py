#encoding: utf-8

import torch
from math import exp, log, sqrt
from torch import nn
from torch.autograd import Function
from torch.utils.cpp_extension import load

from modules.act import Custom_Act, LGLU, get_act, reduce_model as reduce_model_act
from modules.dropout import Dropout, reduce_model as reduce_model_drop
from utils.base import reduce_model_list
from utils.decode.beam import repeat_bsize_for_beam_tensor
from utils.fmt.parser import parse_none
from utils.relpos.bucket import build_rel_pos_bucket, build_rel_pos_bucket_map
from utils.torch.comp import torch_no_grad
from utils.torch.pyc import transfer_CNone_tuple

from cnfg.ihyp import *

Linear = nn.Linear

class PositionwiseFF(nn.Module):

	# isize: input dimension
	# hsize: hidden dimension

	def __init__(self, isize, hsize=None, dropout=0.0, act_drop=None, norm_residual=norm_residual_default, custom_act=use_adv_act_default, enable_bias=enable_prev_ln_bias_default, use_glu=use_glu_ffn, **kwargs):

		super(PositionwiseFF, self).__init__()

		_hsize = isize * 4 if hsize is None else hsize
		_act_drop = parse_none(act_drop, dropout)

		if (use_glu is not None) and (_hsize % 2 == 1):
			_hsize += 1

		_ = [Linear(isize, _hsize)]
		_drop_ind = 2
		if use_glu is None:
			_.extend([Custom_Act() if custom_act else nn.ReLU(inplace=True), Linear(_hsize, isize, bias=enable_bias)])
		else:
			use_glu = use_glu.lower()
			if use_glu == "glu":
				_.append(nn.GLU())
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

		self.normer = nn.LayerNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)

		self.norm_residual = norm_residual

		if self.c_available() and (use_glu is None):
			self.c_init()

	def forward(self, x, **kwargs):

		_out = self.normer(x)

		out = self.net(_out)

		out = out + (_out if self.norm_residual else x)

		return out

	def c_available(self):

		return use_c_backend_pff and (type(self) == PositionwiseFF)

	def c_init(self, bind=bind_c_forward):

		try:
			import pff_cpp
		except Exception as e:
			pff_cpp = load(name="pff_cpp", sources=["modules/cpp/base/ffn/pff.cpp", "modules/cpp/base/ffn/pff_func.cpp", "modules/cpp/act/act_func.cpp"])
		try:
			import act_cpp
		except Exception as e:
			act_cpp = load(name="act_cpp", sources=["modules/cpp/act/act.cpp", "modules/cpp/act/act_func.cpp"])
		self.c_forward_func = pff_cpp.forward
		self.c_act_func = act_cpp.get_func(adv_act if use_adv_act_default else "relu")
		self.c_build_cache()
		if bind:
			PositionwiseFF.forward = PositionwiseFF.c_forward

	def c_forward(self, x):

		return self.c_forward_func(*self.c_build_inputs(x))

	def c_build_cache(self):

		self.bargs = {"net.1.inplace": (self.net[1].inplace if hasattr(self.net[1], "inplace") else False), "norm_residual": self.norm_residual}
		dargs = {"normer.eps": self.normer.eps}
		if len(self.net) > 3:
			self.bargs["net.2.inplace"] = self.net[2].inplace
			self.bargs["net.4.inplace"] = self.net[4].inplace
			dargs["net.2.p"] = self.net[2].p
		else:
			dargs["net.2.p"] = 0.0
		self.aargs = (self.c_act_func, dargs, self.normer.normalized_shape)
		self.targs = dict(self.named_parameters())

	def c_build_inputs(self, x):

		i_d = self.targs.copy()
		i_d["x"] = x
		if len(self.net) > 3:
			bargs = self.bargs.copy()
			bargs["net.2.training"] = self.net[2].training
			bargs["net.4.training"] = self.net[4].training
		else:
			bargs = self.bargs

		return i_d, bargs, *self.aargs

class PositionalEmb(nn.Module):

	# num_dim: dimension of embedding
	# num_pos: maximum length of sentence cached, extended length will be generated while needed and droped immediately after that
	# pos_offset: initial offset for position
	# dim_offset: initial offset for dimension

	def __init__(self, num_dim, num_pos=cache_len_default, pos_offset=0, dim_offset=0, alpha=1.0, **kwargs):

		super(PositionalEmb, self).__init__()

		self.num_pos = num_pos
		self.num_dim = num_dim
		self.poff = pos_offset
		self.doff = dim_offset
		self.alpha = alpha
		self.register_buffer("w", torch.Tensor(num_pos, num_dim), persistent=False)
		self.reset_parameters()

	# x: input (bsize, seql)

	def forward(self, x, expand=True, **kwargs):

		bsize, seql = x.size()

		rs = self.w[:seql].unsqueeze(0) if seql <= self.num_pos else torch.cat((self.w, self.get_ext(seql, False)), 0).unsqueeze(0)

		return rs.expand(bsize, seql, self.num_dim) if expand else rs

	def reset_parameters(self):

		poff = self.poff
		pos = torch.arange(poff, self.num_pos + poff, dtype=self.w.dtype, device=self.w.device).unsqueeze(1)
		rdiv_term = (torch.arange(self.doff, self.num_dim + self.doff, 2, dtype=self.w.dtype, device=self.w.device) * -(log(1e4) / self.num_dim)).exp()
		_tmp = pos * rdiv_term
		if self.alpha != 1.0:
			_tmp.mul_(self.alpha)
		self.w[:, 0::2], self.w[:, 1::2] = _tmp.sin(), (_tmp.narrow(-1, 0, _tmp.size(-1) - 1).cos() if self.num_dim % 2 == 1 else _tmp.cos())

	def get_ext(self, length, step_pick=False):

		poff = self.poff

		if step_pick:
			pos = torch.as_tensor([length + poff], dtype=self.w.dtype, device=self.w.device).unsqueeze(1)
			ed = self.w.new_empty(1, self.num_dim)
		else:
			npos = self.num_pos
			pos = torch.arange(npos + poff, length + poff, dtype=self.w.dtype, device=self.w.device).unsqueeze(1)
			ed = self.w.new_empty(length - npos, self.num_dim)
		rdiv_term = (torch.arange(self.doff, self.num_dim + self.doff, 2, dtype=self.w.dtype, device=self.w.device) * -(log(1e4) / self.num_dim)).exp()
		_tmp = pos * rdiv_term
		if self.alpha != 1.0:
			_tmp.mul_(self.alpha)
		ed[:, 0::2], ed[:, 1::2] = _tmp.sin(), (_tmp.narrow(-1, 0, _tmp.size(-1) - 1).cos() if self.num_dim % 2 == 1 else _tmp.cos())

		return ed

	# step of weight to retrieve, start from 0

	def get_pos(self, step):

		return self.w[step] if step < self.num_pos else self.get_ext(step, True).squeeze(0)

class MultiHeadAttn(nn.Module):

	# isize: input dimension
	# hsize: hidden dimension
	# osize: output size of this layer
	# num_head: number of heads
	# dropout: dropout probability
	# k_rel_pos: uni-directional window size of relative positional encoding
	# uni_direction_reduction: performing resource reduction for uni-directional self-attention
	# is_left_to_right_reduction: only for uni_direction_reduction, indicating left-to-right self-attention or right-to-left
	# zero_reduction: only for uni_direction_reduction, using zeros for padding positions in the relative positional matrix
	# sparsenorm: using sparse normer or standard softmax
	# bind_qk: query and key can share a same linear transformation for the Reformer: The Efficient Transformer (https://arxiv.org/abs/2001.04451) paper.

	def __init__(self, isize, hsize, osize, num_head=8, dropout=0.0, k_isize=None, v_isize=None, enable_bias=enable_prev_ln_bias_default, enable_proj_bias=enable_proj_bias_default, k_rel_pos=0, uni_direction_reduction=False, is_left_to_right_reduction=True, zero_reduction=relpos_reduction_with_zeros, max_bucket_distance=0, sparsenorm=False, bind_qk=False, xseql=cache_len_default, is_decoding=False, **kwargs):

		super(MultiHeadAttn, self).__init__()

		self.attn_dim = hsize // num_head
		self.hsize = self.attn_dim * num_head
		self.num_head = num_head

		self.query_adaptor = Linear(isize, self.hsize, bias=enable_proj_bias)
		_k_isize = parse_none(k_isize, isize)
		self.key_adaptor = self.query_adaptor if bind_qk and isize == _k_isize else Linear(_k_isize, self.hsize, bias=enable_proj_bias)
		self.value_adaptor = Linear(_k_isize if v_isize is None else v_isize, self.hsize, bias=enable_proj_bias)

		self.outer = Linear(self.hsize, osize, bias=enable_bias)

		#self.normer = MHSparseNormer(num_head, dim=-1) if sparsenorm else nn.Softmax(dim=-1)
		self.normer = SparseNormer(dim=-1) if sparsenorm else nn.Softmax(dim=-1)

		self.drop = Dropout(dropout, inplace=sparsenorm) if dropout > 0.0 else None

		if k_rel_pos > 0:
			self.rel_shift = k_rel_pos
			if max_bucket_distance > 0:
				self.register_buffer("rel_pos_map", build_rel_pos_bucket_map(k_rel_pos=k_rel_pos, max_len=max_bucket_distance, uni_direction=uni_direction_reduction), persistent=False)
				self.register_buffer("rel_pos", build_rel_pos_bucket(xseql, k_rel_pos=k_rel_pos, max_len=max_bucket_distance, uni_direction=uni_direction_reduction, dis_map=self.rel_pos_map), persistent=False)
				self.rel_pemb = nn.Embedding((k_rel_pos + 1) if uni_direction else (k_rel_pos + k_rel_pos + 1), self.num_head)
				self.clamp_max, self.clamp_min = max_bucket_distance, uni_direction_reduction
			else:
				padding_idx = None
				if uni_direction_reduction:
					_n_pemb = k_rel_pos + 1
					if is_left_to_right_reduction:
						self.clamp_min, self.clamp_max = -k_rel_pos, 0,
					else:
						self.clamp_min, self.clamp_max, self.rel_shift = 0, k_rel_pos, 0
					if zero_reduction:
						_n_pemb += 1
						if is_left_to_right_reduction:
							self.clamp_max += 1
							padding_idx = self.clamp_max
						else:
							self.clamp_min -= 1
							self.rel_shift += 1
							padding_idx = 0
				else:
					_n_pemb = k_rel_pos + k_rel_pos + 1
					self.clamp_min, self.clamp_max = -k_rel_pos, k_rel_pos
				self.rel_pemb = nn.Embedding(_n_pemb, self.attn_dim, padding_idx=padding_idx)
				_rpm = torch.arange(0, xseql, dtype=torch.long)
				self.register_buffer("rel_pos", (_rpm.unsqueeze(0) - _rpm.unsqueeze(1)).clamp(min=self.clamp_min, max=self.clamp_max) + self.rel_shift, persistent=False)
				self.register_buffer("rel_pos_map", None, persistent=False)
			self.xseql = xseql
			# the buffer can be shared inside the encoder or the decoder across layers for saving memory, by setting self.ref_rel_posm of self attns in deep layers to SelfAttn in layer 0, and sharing corresponding self.rel_pos
			self.ref_rel_posm = None
			self.register_buffer("rel_pos_cache", None, persistent=False)
		else:
			self.rel_pemb = None

		self.register_buffer("real_iK", None, persistent=False)
		self.register_buffer("real_iV", None, persistent=False)
		self.register_buffer("iK", None, persistent=False)
		self.register_buffer("iV", None, persistent=False)
		self.is_decoding = is_decoding

		if self.c_available():
			self.c_init()

	# iQ: query (bsize, num_query, vsize)
	# iK: keys (bsize, seql, vsize)
	# iV: values (bsize, seql, vsize)
	# mask (bsize, num_query, seql)

	def forward(self, iQ, iK, iV, mask=None, states=None, **kwargs):

		bsize, nquery = iQ.size()[:2]
		seql = iK.size(1)
		nheads = self.num_head
		adim = self.attn_dim

		# real_iQ: MultiHead iQ (bsize, num_query, vsize) => (bsize, nheads, nquery, adim)
		# real_iK: MultiHead iK (bsize, seql, vsize) => (bsize, nheads, adim, seql)
		# real_iV: MultiHead iV (bsize, seql, vsize) => (bsize, nheads, seql, adim)

		real_iQ = self.query_adaptor(iQ).view(bsize, nquery, nheads, adim).transpose(1, 2)

		if (self.real_iK is not None) and self.iK.is_set_to(iK) and self.is_decoding:
			real_iK = self.real_iK
		else:
			real_iK = self.key_adaptor(iK).view(bsize, seql, nheads, adim).permute(0, 2, 3, 1)
			if self.is_decoding:
				self.iK, self.real_iK = iK, real_iK
		if (self.real_iV is not None) and self.iV.is_set_to(iV) and (not self.training):
			real_iV = self.real_iV
		else:
			real_iV = self.value_adaptor(iV).view(bsize, seql, nheads, adim).transpose(1, 2)
			if not self.training:
				self.iV, self.real_iV = iV, real_iV

		if states is not None:
			_h_real_iK, _h_real_iV = states
			if _h_real_iK is not None:
				seql += _h_real_iK.size(-1)
				real_iK, real_iV = torch.cat((_h_real_iK, real_iK,), dim=-1), torch.cat((_h_real_iV, real_iV,), dim=2)

		# scores (bsize, nheads, nquery, adim) * (bsize, nheads, adim, seql) => (bsize, nheads, nquery, seql)

		scores = real_iQ.matmul(real_iK)

		if self.rel_pemb is not None:
			self.rel_pos_cache = self.get_rel_pos(seql).narrow(0, seql - nquery, nquery).contiguous() if self.ref_rel_posm is None else self.ref_rel_posm.rel_pos_cache
			scores += real_iQ.permute(2, 0, 1, 3).contiguous().view(nquery, bsize * nheads, adim).bmm(self.rel_pemb(self.rel_pos_cache).transpose(1, 2)).view(nquery, bsize, nheads, seql).permute(1, 2, 0, 3) if self.rel_pos_map is None else self.rel_pemb(self.rel_pos_cache).permute(2, 0, 1)

		scores = scores / sqrt(adim)

		if mask is not None:
			scores.masked_fill_(mask.unsqueeze(1), -inf_default)

		scores = self.normer(scores)

		if self.drop is not None:
			scores = self.drop(scores)

		# output of this layer T((bsize, nheads, nquery, seql) * (bsize, nheads, seql, adim)) => (bsize, nquery, nheads, adim) => (bsize, nquery, osize)

		out = self.outer(scores.matmul(real_iV).transpose(1, 2).contiguous().view(bsize, nquery, self.hsize))

		if states is None:
			return out
		else:
			return out, (real_iK, real_iV,)

	def train(self, mode=True):

		super(MultiHeadAttn, self).train(mode)
		if mode:
			self.reset_buffer()

		return self

	def get_rel_pos(self, length):

		if length <= self.xseql:
			return self.rel_pos.narrow(0, 0, length).narrow(1, 0, length)
		else:
			if self.rel_pos_map is None:
				_rpm = torch.arange(0, length, dtype=self.rel_pos.dtype, device=self.rel_pos.device)
				return ((_rpm.unsqueeze(0) - _rpm.unsqueeze(1)).clamp(min=self.clamp_min, max=self.clamp_max) + self.rel_shift)
			else:
				return build_rel_pos_bucket(length, k_rel_pos=self.rel_shift, max_len=self.clamp_max, uni_direction=self.clamp_min, device=self.rel_pos.device, dis_map=self.rel_pos_map)

	def reset_buffer(self, value=None):

		self.iK = self.iV = self.real_iK = self.real_iV = self.rel_pos_cache = value

	def repeat_buffer(self, beam_size):

		if self.real_iK is not None:
			self.real_iK = repeat_bsize_for_beam_tensor(self.real_iK, beam_size)
		if self.real_iV is not None:
			self.real_iV = repeat_bsize_for_beam_tensor(self.real_iV, beam_size)

	def index_buffer(self, indices, dim=0):

		if self.real_iK is not None:
			self.real_iK = self.real_iK.index_select(dim, indices)
		if self.real_iV is not None:
			self.real_iV = self.real_iV.index_select(dim, indices)

	def c_available(self):

		return use_c_backend_mhattn and (type(self) == MultiHeadAttn) and (type(self.normer) == nn.Softmax) and ((self.rel_pos is None) or (self.rel_pos_map is None))

	def c_init(self, bind=bind_c_forward):

		try:
			import attn_cpp
		except Exception as e:
			attn_cpp = load(name="attn_cpp", sources=["modules/cpp/base/attn/attn.cpp"])
		self.c_forward_func = attn_cpp.forward
		self.c_build_cache()
		if bind:
			MultiHeadAttn.forward = MultiHeadAttn.c_forward

	def c_forward(self, iQ, iK, iV, mask=None, states=None):

		return self.c_process_output(self.c_forward_func(*self.c_build_inputs(iQ, iK, iV, mask=mask, states=states)), iK, iV, states=states)

	def c_build_cache(self):

		iargs = {"num_head": self.num_head, "attn_dim": self.attn_dim}
		if self.rel_pemb is not None:
			iargs.update({"rel_pemb.padding_idx": self.rel_pemb.padding_idx, "clamp_min": self.clamp_min, "clamp_max": self.clamp_max, "rel_shift": self.rel_shift})
		self.aargs = (iargs, 0.0 if self.drop is None else self.drop.p, inf_default,)
		self.targs = dict(self.named_parameters())

	def c_build_inputs(self, iQ, iK, iV, mask=None, states=None):

		i_d = self.targs.copy()
		i_d.update({"iQ": iQ, "iK":iK, "iV": iV})
		if mask is not None:
			i_d["mask"] = mask
		_buf_d = dict(self.named_buffers())
		if "iK" in _buf_d:
			_buf_d["buf_iK"] = _buf_d.pop("iK")
		if "iV" in _buf_d:
			_buf_d["buf_iV"] = _buf_d.pop("iV")
		i_d.update(_buf_d)
		if self.rel_pemb is not None:
			if self.ref_rel_posm is not None:
				i_d["rel_pos_cache"] = self.ref_rel_posm.rel_pos_cache

		return i_d, [] if states is None else transfer_CNone_tuple(states), *self.aargs, {"drop.inplace": False if self.drop is None else self.drop.inplace, "training": self.training, "drop.training": self.training if self.drop is None else self.drop.training}

	def c_process_output(self, rs, iK, iV, states=None):

		if self.rel_pemb is not None:
			self.rel_pos_cache = rs["rel_pos_cache"]

		evaluation = not self.training
		if (states is not None) or evaluation:
			real_iK, real_iV = rs["real_iK"], rs["real_iV"]
			if evaluation:
				self.iK, self.real_iK, self.iV, self.real_iV = iK, real_iK, iV, real_iV

		if states is None:
			return rs["out"]
		else:
			return rs["out"], (real_iK, real_iV,)

# Accelerated MultiHeadAttn for self attention, use when Q == K == V
class SelfAttn(nn.Module):

	def __init__(self, isize, hsize, osize, num_head=8, dropout=0.0, enable_bias=enable_prev_ln_bias_default, enable_proj_bias=enable_proj_bias_default, k_rel_pos=use_k_relative_position, uni_direction_reduction=False, is_left_to_right_reduction=True, zero_reduction=relpos_reduction_with_zeros, max_bucket_distance=0, sparsenorm=False, xseql=cache_len_default, **kwargs):

		super(SelfAttn, self).__init__()

		self.attn_dim = hsize // num_head
		self.hsize = self.attn_dim * num_head
		self.num_head = num_head

		self.adaptor = Linear(isize, self.hsize * 3, bias=enable_proj_bias)

		self.outer = Linear(self.hsize, osize, bias=enable_bias)

		#self.normer = MHSparseNormer(num_head, dim=-1) if sparsenorm else nn.Softmax(dim=-1)
		self.normer = SparseNormer(dim=-1) if sparsenorm else nn.Softmax(dim=-1)

		self.drop = Dropout(dropout, inplace=sparsenorm) if dropout > 0.0 else None

		if k_rel_pos > 0:
			self.rel_shift = k_rel_pos
			if max_bucket_distance > 0:
				self.register_buffer("rel_pos_map", build_rel_pos_bucket_map(k_rel_pos=k_rel_pos, max_len=max_bucket_distance, uni_direction=uni_direction_reduction), persistent=False)
				self.register_buffer("rel_pos", build_rel_pos_bucket(xseql, k_rel_pos=k_rel_pos, max_len=max_bucket_distance, uni_direction=uni_direction_reduction, dis_map=self.rel_pos_map), persistent=False)
				self.rel_pemb = nn.Embedding((k_rel_pos + 1) if uni_direction_reduction else (k_rel_pos + k_rel_pos + 1), self.num_head)
				self.clamp_max, self.clamp_min = max_bucket_distance, uni_direction_reduction
			else:
				padding_idx = None
				if uni_direction_reduction:
					_n_pemb = k_rel_pos + 1
					if is_left_to_right_reduction:
						self.clamp_min, self.clamp_max = -k_rel_pos, 0,
					else:
						self.clamp_min, self.clamp_max, self.rel_shift = 0, k_rel_pos, 0
					if zero_reduction:
						_n_pemb += 1
						if is_left_to_right_reduction:
							self.clamp_max += 1
							padding_idx = self.clamp_max
						else:
							self.clamp_min -= 1
							self.rel_shift += 1
							padding_idx = 0
				else:
					_n_pemb = k_rel_pos + k_rel_pos + 1
					self.clamp_min, self.clamp_max = -k_rel_pos, k_rel_pos
				self.rel_pemb = nn.Embedding(_n_pemb, self.attn_dim, padding_idx=padding_idx)
				_rpm = torch.arange(0, xseql, dtype=torch.long)
				self.register_buffer("rel_pos", (_rpm.unsqueeze(0) - _rpm.unsqueeze(1)).clamp(min=self.clamp_min, max=self.clamp_max) + self.rel_shift, persistent=False)
				self.register_buffer("rel_pos_map", None, persistent=False)
			self.xseql = xseql
			# the buffer can be shared inside the encoder or the decoder across layers for saving memory, by setting self.ref_rel_posm of self attns in deep layers to SelfAttn in layer 0, and sharing corresponding self.rel_pos
			self.ref_rel_posm = None
			self.register_buffer("rel_pos_cache", None, persistent=False)
		else:
			self.rel_pemb = None

		if self.c_available():
			self.c_init()

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

		if self.rel_pemb is not None:
			if states is None:
				self.rel_pos_cache = self.get_rel_pos(nquery).contiguous() if self.ref_rel_posm is None else self.ref_rel_posm.rel_pos_cache
				scores += real_iQ.permute(2, 0, 1, 3).contiguous().view(nquery, bsize * nheads, adim).bmm(self.rel_pemb(self.rel_pos_cache).transpose(1, 2)).view(nquery, bsize, nheads, nquery).permute(1, 2, 0, 3) if self.rel_pos_map is None else self.rel_pemb(self.rel_pos_cache).permute(2, 0, 1)
			else:
				self.rel_pos_cache = self.get_rel_pos(seql).narrow(0, seql - nquery, nquery).contiguous() if self.ref_rel_posm is None else self.ref_rel_posm.rel_pos_cache
				scores += real_iQ.permute(2, 0, 1, 3).contiguous().view(nquery, bsize * nheads, adim).bmm(self.rel_pemb(self.rel_pos_cache).transpose(1, 2)).view(nquery, bsize, nheads, seql).permute(1, 2, 0, 3) if self.rel_pos_map is None else self.rel_pemb(self.rel_pos_cache).permute(2, 0, 1)

		scores = scores / sqrt(adim)

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

	def get_rel_pos(self, length):

		if length <= self.xseql:
			return self.rel_pos.narrow(0, 0, length).narrow(1, 0, length)
		else:
			if self.rel_pos_map is None:
				_rpm = torch.arange(0, length, dtype=self.rel_pos.dtype, device=self.rel_pos.device)
				return ((_rpm.unsqueeze(0) - _rpm.unsqueeze(1)).clamp(min=self.clamp_min, max=self.clamp_max) + self.rel_shift)
			else:
				return build_rel_pos_bucket(length, k_rel_pos=self.rel_shift, max_len=self.clamp_max, uni_direction=self.clamp_min, device=self.rel_pos.device, dis_map=self.rel_pos_map)

	def reset_buffer(self, value=None):

		self.rel_pos_cache = value

	def c_available(self):

		return use_c_backend_selfattn and (type(self) == SelfAttn) and (type(self.normer) == nn.Softmax) and ((self.rel_pos is None) or (self.rel_pos_map is None))

	def c_init(self, bind=bind_c_forward):

		try:
			import self_attn_cpp
		except Exception as e:
			self_attn_cpp = load(name="self_attn_cpp", sources=["modules/cpp/base/attn/self/attn.cpp"])
		self.c_forward_func = self_attn_cpp.forward
		self.c_build_cache()
		if bind:
			SelfAttn.forward = SelfAttn.c_forward

	def c_forward(self, iQ, mask=None, states=None):

		return self.c_process_output(self.c_forward_func(*self.c_build_inputs(iQ, mask=mask, states=states)), states=states)

	def c_build_cache(self):

		iargs = {"num_head": self.num_head, "attn_dim": self.attn_dim}
		if self.rel_pemb is not None:
			iargs.update({"rel_pemb.padding_idx": self.rel_pemb.padding_idx, "clamp_min": self.clamp_min, "clamp_max": self.clamp_max, "rel_shift": self.rel_shift})
		self.aargs = (iargs, 0.0 if self.drop is None else self.drop.p, inf_default,)
		self.targs = dict(self.named_parameters())
		self.targs.update(self.named_buffers())

	def c_build_inputs(self, iQ, mask=None, states=None):

		i_d = self.targs.copy()
		i_d["iQ"] = iQ
		if mask is not None:
			i_d["mask"] = mask
		if self.rel_pemb is not None:
			if self.ref_rel_posm is not None:
				i_d["rel_pos_cache"] = self.ref_rel_posm.rel_pos_cache

		return i_d, [] if states is None else transfer_CNone_tuple(states), *self.aargs, {"drop.inplace": False if self.drop is None else self.drop.inplace, "drop.training": self.training if self.drop is None else self.drop.training}

	def c_process_output(self, rs, states=None):

		if self.rel_pemb is not None:
			self.rel_pos_cache = rs["rel_pos_cache"]

		if states is None:
			return rs["out"]
		else:
			return rs["out"], (rs["real_iK"], rs["real_iV"],)

# Accelerated MultiHeadAttn for cross attention, use when K == V
class CrossAttn(nn.Module):

	def __init__(self, isize, hsize, osize, num_head=8, dropout=0.0, k_isize=None, enable_bias=enable_prev_ln_bias_default, enable_proj_bias=enable_proj_bias_default, sparsenorm=False, is_decoding=False, **kwargs):

		super(CrossAttn, self).__init__()

		self.attn_dim = hsize // num_head
		self.hsize = self.attn_dim * num_head
		self.num_head = num_head

		self.query_adaptor = Linear(isize, self.hsize, bias=enable_proj_bias)

		self.kv_adaptor = Linear(isize if k_isize is None else k_isize, self.hsize * 2, bias=enable_proj_bias)

		self.outer = Linear(self.hsize, osize, bias=enable_bias)

		#self.normer = MHSparseNormer(num_head, dim=-1) if sparsenorm else nn.Softmax(dim=-1)
		self.normer = SparseNormer(dim=-1) if sparsenorm else nn.Softmax(dim=-1)

		self.drop = Dropout(dropout, inplace=sparsenorm) if dropout > 0.0 else None

		self.register_buffer("real_iK", None, persistent=False)
		self.register_buffer("real_iV", None, persistent=False)
		self.register_buffer("iK", None, persistent=False)
		self.is_decoding = is_decoding

		if self.c_available():
			self.c_init()

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
			real_iK, real_iV = real_iK.permute(0, 2, 3, 1), real_iV.transpose(1, 2)
			if self.is_decoding:
				self.iK, self.real_iK, self.real_iV = iK, real_iK, real_iV

		scores = real_iQ.matmul(real_iK) / sqrt(adim)

		if mask is not None:
			scores.masked_fill_(mask.unsqueeze(1), -inf_default)

		scores = self.normer(scores)

		if self.drop is not None:
			scores = self.drop(scores)

		return self.outer(scores.matmul(real_iV).transpose(1, 2).contiguous().view(bsize, nquery, self.hsize))

	def train(self, mode=True):

		super(CrossAttn, self).train(mode)

		if mode:
			self.reset_buffer()
		self.is_decoding = not mode

		return self

	def reset_buffer(self, value=None):

		self.iK = self.real_iK = self.real_iV = value

	def repeat_buffer(self, beam_size):

		if self.real_iK is not None:
			self.real_iK, self.real_iV = repeat_bsize_for_beam_tensor(self.real_iK, beam_size), repeat_bsize_for_beam_tensor(self.real_iV, beam_size)

	def index_buffer(self, indices, dim=0):

		if self.real_iK is not None:
			self.real_iK, self.real_iV = self.real_iK.index_select(dim, indices), self.real_iV.index_select(dim, indices)

	def c_available(self):

		return use_c_backend_crossattn and (type(self) == CrossAttn) and (type(self.normer) == nn.Softmax)

	def c_init(self, bind=bind_c_forward):

		try:
			import cross_attn_cpp
		except Exception as e:
			cross_attn_cpp = load(name="cross_attn_cpp", sources=["modules/cpp/base/attn/cross/attn.cpp"])
		self.c_forward_func = cross_attn_cpp.forward
		self.c_build_cache()
		if bind:
			CrossAttn.forward = CrossAttn.c_forward

	def c_forward(self, iQ, iK, mask=None):

		return self.c_process_output(self.c_forward_func(*self.c_build_inputs(iQ, iK, mask=mask)), iK)

	def c_build_cache(self):

		self.aargs = ({"num_head": self.num_head, "attn_dim": self.attn_dim}, 0.0 if self.drop is None else self.drop.p, inf_default,)
		self.targs = dict(self.named_parameters())

	def c_build_inputs(self, iQ, iK, mask=None):

		i_d = self.targs.copy()
		i_d["iQ"] = iQ
		i_d["iK"] = iK
		if mask is not None:
			i_d["mask"] = mask
		i_d.update(self.named_parameters())
		_buf_d = dict(self.named_buffers())
		if "iK" in _buf_d:
			_buf_d["buf_iK"] = _buf_d.pop("iK")
		i_d.update(_buf_d)

		return i_d, *self.aargs, {"drop.inplace": False if self.drop is None else self.drop.inplace, "training": self.training, "drop.training": self.training if self.drop is None else self.drop.training}

	def c_process_output(self, rs, iK):

		if not self.training:
			self.iK, self.real_iK, self.real_iV = iK, rs["real_iK"], rs["real_iV"]

		return rs["out"]

class ResMHAttn(nn.Module):

	def __init__(self, isize, hsize, num_head=8, dropout=0.0, norm_residual=norm_residual_default, **kwargs):

		super(ResMHAttn, self).__init__()

		self.net = MultiHeadAttn(isize, hsize, isize, num_head=num_head, dropout=dropout, **kwargs)
		self.normer = nn.LayerNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)
		self.drop = Dropout(dropout, inplace=True) if dropout > 0.0 else None
		self.norm_residual = norm_residual

		if self.c_available():
			self.c_init()

	def forward(self, iQ, iK, iV, *inputs, **kwargs):

		_iQ = self.normer(iQ)
		_iK = _iQ if iK.is_set_to(iQ) else iK
		_iV = _iK if iV.is_set_to(iK) else iV

		outs = self.net(_iQ, _iK, _iV, *inputs, **kwargs)

		if isinstance(outs, tuple):
			_out = outs[0]

			if self.drop is not None:
				_out = self.drop(_out)

			return _out + (_iQ if self.norm_residual else iQ), *outs[1:]

		else:
			if self.drop is not None:
				outs = self.drop(outs)

			return outs + (_iQ if self.norm_residual else iQ)

	def load_base(self, base_module):

		self.normer, self.drop, self.norm_residual = base_module.normer, base_module.drop, base_module.norm_residual
		if hasattr(self.net, "load_base"):
			self.net.load_base(base_module.net)
		else:
			self.net = base_module.net

	def c_available(self):

		return use_c_backend_mhattn and (type(self) == ResMHAttn) and self.net.c_available()

	def c_init(self, bind=bind_c_forward):

		try:
			import res_attn_cpp
		except Exception as e:
			res_attn_cpp = load(name="res_attn_cpp", sources=["modules/cpp/base/resattn/attn.cpp"])
		self.c_forward_func = res_attn_cpp.forward
		self.c_build_cache()
		if bind:
			ResMHAttn.forward = ResMHAttn.c_forward

	def c_forward(self, iQ, iK, iV, mask=None, states=None):

		return self.c_process_output(self.c_forward_func(*self.c_build_inputs(iQ, iK, iV, mask=mask, states=states)), iK, iV, states=states)

	def c_build_cache(self):

		iargs = {"net.num_head": self.net.num_head, "net.attn_dim": self.net.attn_dim}
		if self.net.rel_pemb is not None:
			iargs.update({"net.rel_pemb.padding_idx": self.net.rel_pemb.padding_idx, "net.clamp_min": self.net.clamp_min, "net.clamp_max": self.net.clamp_max, "net.rel_shift": self.net.rel_shift})
		self.aargs = (iargs, {"normer.eps": self.normer.eps, "inf_value": inf_default, "net.drop.p": 0.0 if self.net.drop is None else self.net.drop.p, "drop.p": 0.0 if self.drop is None else self.drop.p}, self.normer.normalized_shape,)
		self.bargs = {"net.drop.inplace": False if self.net.drop is None else self.net.drop.inplace, "drop.inplace": False if self.drop is None else self.drop.inplace, "norm_residual": self.norm_residual}
		self.targs = dict(self.named_parameters())

	def c_build_inputs(self, iQ, iK, iV, mask=None, states=None):

		i_d = self.targs.copy()
		i_d.update({"iQ": iQ, "iK":iK, "iV": iV})
		if mask is not None:
			i_d["mask"] = mask
		i_d.update(self.named_buffers())
		if self.net.rel_pemb is not None:
			if self.net.ref_rel_posm is not None:
				i_d["net.rel_pos_cache"] = self.net.ref_rel_posm.rel_pos_cache
		bargs = self.bargs.copy()
		bargs.update({"net.training": self.net.training, "net.drop.training": self.net.training if self.net.drop is None else self.net.drop.training, "drop.training": self.training if self.drop is None else self.drop.training})

		return i_d, [] if states is None else transfer_CNone_tuple(states), *self.aargs, bargs

	def c_process_output(self, rs, iK, iV, states=None):

		if self.net.rel_pemb is not None:
			self.net.rel_pos_cache = rs["net.rel_pos_cache"]

		evaluation = not self.net.training
		if (states is not None) or evaluation:
			real_iK, real_iV = rs["net.real_iK"], rs["net.real_iV"]
			if evaluation:
				self.net.iK, self.net.real_iK, self.net.iV, self.net.real_iV = iK, real_iK, iV, real_iV

		if states is None:
			return rs["out"]
		else:
			return rs["out"], (real_iK, real_iV,)

class ResSelfAttn(nn.Module):

	def __init__(self, isize, hsize, num_head=8, dropout=0.0, norm_residual=norm_residual_default, **kwargs):

		super(ResSelfAttn, self).__init__()

		self.net = SelfAttn(isize, hsize, isize, num_head=num_head, dropout=dropout, **kwargs)
		self.normer = nn.LayerNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)
		self.drop = Dropout(dropout, inplace=True) if dropout > 0.0 else None
		self.norm_residual = norm_residual

		if self.c_available():
			self.c_init()

	def forward(self, iQ, *inputs, **kwargs):

		_iQ = self.normer(iQ)

		outs = self.net(_iQ, *inputs, **kwargs)

		if isinstance(outs, tuple):
			_out = outs[0]

			if self.drop is not None:
				_out = self.drop(_out)

			return _out + (_iQ if self.norm_residual else iQ), *outs[1:]

		else:
			if self.drop is not None:
				outs = self.drop(outs)

			return outs + (_iQ if self.norm_residual else iQ)

	def load_base(self, base_module):

		self.normer, self.drop, self.norm_residual = base_module.normer, base_module.drop, base_module.norm_residual
		if hasattr(self.net, "load_base"):
			self.net.load_base(base_module.net)
		else:
			self.net = base_module.net

	def c_available(self):

		return use_c_backend_selfattn and (type(self) == ResSelfAttn) and self.net.c_available()

	def c_init(self, bind=bind_c_forward):

		try:
			import res_self_attn_cpp
		except Exception as e:
			res_self_attn_cpp = load(name="res_self_attn_cpp", sources=["modules/cpp/base/resattn/self/attn.cpp"])
		self.c_forward_func = res_self_attn_cpp.forward
		self.c_build_cache()
		if bind:
			ResSelfAttn.forward = ResSelfAttn.c_forward

	def c_forward(self, iQ, mask=None, states=None):

		return self.c_process_output(self.c_forward_func(*self.c_build_inputs(iQ, mask=mask, states=states)), states=states)

	def c_build_cache(self):

		iargs = {"net.num_head": self.net.num_head, "net.attn_dim": self.net.attn_dim}
		if self.net.rel_pemb is not None:
			iargs.update({"net.rel_pemb.padding_idx": self.net.rel_pemb.padding_idx, "net.clamp_min": self.net.clamp_min, "net.clamp_max": self.net.clamp_max, "net.rel_shift": self.net.rel_shift})
		self.aargs = (iargs, {"normer.eps": self.normer.eps, "inf_value": inf_default, "net.drop.p": 0.0 if self.net.drop is None else self.net.drop.p, "drop.p": 0.0 if self.drop is None else self.drop.p}, self.normer.normalized_shape,)
		self.bargs = {"net.drop.inplace": False if self.net.drop is None else self.net.drop.inplace, "drop.inplace": False if self.drop is None else self.drop.inplace, "norm_residual": self.norm_residual}
		self.targs = dict(self.named_parameters())

	def c_build_inputs(self, iQ, mask=None, states=None):

		i_d = self.targs.copy()
		i_d["iQ"] = iQ
		if mask is not None:
			i_d["mask"] = mask
		if self.net.rel_pemb is not None:
			if self.net.ref_rel_posm is not None:
				i_d["net.rel_pos_cache"] = self.net.ref_rel_posm.rel_pos_cache
		bargs = self.bargs.copy()
		bargs.update({"net.training": self.net.training, "net.drop.training": self.net.training if self.net.drop is None else self.net.drop.training, "drop.training": self.training if self.drop is None else self.drop.training})

		return i_d, [] if states is None else transfer_CNone_tuple(states), *self.aargs, bargs

	def c_process_output(self, rs, states=None):

		if self.net.rel_pemb is not None:
			self.net.rel_pos_cache = rs["net.rel_pos_cache"]

		if states is None:
			return rs["out"]
		else:
			return rs["out"], (rs["net.real_iK"], rs["net.real_iV"],)

class ResCrossAttn(nn.Module):

	def __init__(self, isize, hsize, num_head=8, dropout=0.0, norm_residual=norm_residual_default, **kwargs):

		super(ResCrossAttn, self).__init__()

		self.net = CrossAttn(isize, hsize, isize, num_head=num_head, dropout=dropout, **kwargs)
		self.normer = nn.LayerNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)
		self.drop = Dropout(dropout, inplace=True) if dropout > 0.0 else None
		self.norm_residual = norm_residual

		if self.c_available():
			self.c_init()

	def forward(self, iQ, iK, *inputs, **kwargs):

		_iQ = self.normer(iQ)

		outs = self.net(_iQ, iK, *inputs, **kwargs)

		if isinstance(outs, tuple):
			_out = outs[0]

			if self.drop is not None:
				_out = self.drop(_out)

			return _out + (_iQ if self.norm_residual else iQ), *outs[1:]

		else:
			if self.drop is not None:
				outs = self.drop(outs)

			return outs + (_iQ if self.norm_residual else iQ)

	def load_base(self, base_module):

		self.normer, self.drop, self.norm_residual = base_module.normer, base_module.drop, base_module.norm_residual
		if hasattr(self.net, "load_base"):
			self.net.load_base(base_module.net)
		else:
			self.net = base_module.net

	def c_available(self):

		return use_c_backend_crossattn and (type(self) == ResCrossAttn) and self.net.c_available()

	def c_init(self, bind=bind_c_forward):

		try:
			import res_cross_attn_cpp
		except Exception as e:
			res_cross_attn_cpp = load(name="res_cross_attn_cpp", sources=["modules/cpp/base/resattn/cross/attn.cpp"])
		self.c_forward_func = res_cross_attn_cpp.forward
		self.c_build_cache()
		if bind:
			ResCrossAttn.forward = ResCrossAttn.c_forward

	def c_forward(self, iQ, iK, mask=None):

		return self.c_process_output(self.c_forward_func(*self.c_build_inputs(iQ, iK, mask=mask)), iK)

	def c_build_cache(self):

		iargs = {"net.num_head": self.net.num_head, "net.attn_dim": self.net.attn_dim}
		self.aargs = (iargs, {"normer.eps": self.normer.eps, "inf_value": inf_default, "net.drop.p": 0.0 if self.net.drop is None else self.net.drop.p, "drop.p": 0.0 if self.drop is None else self.drop.p}, self.normer.normalized_shape,)
		self.bargs = {"net.drop.inplace": False if self.net.drop is None else self.net.drop.inplace, "drop.inplace": False if self.drop is None else self.drop.inplace, "norm_residual": self.norm_residual}
		self.targs = dict(self.named_parameters())

	def c_build_inputs(self, iQ, iK, mask=None):

		i_d = self.targs.copy()
		i_d["iQ"] = iQ
		i_d["iK"] = iK
		if mask is not None:
			i_d["mask"] = mask
		i_d.update(self.named_buffers())
		bargs = self.bargs.copy()
		bargs.update({"net.training": self.net.training, "net.drop.training": self.net.training if self.net.drop is None else self.net.drop.training, "drop.training": self.training if self.drop is None else self.drop.training})

		return i_d, *self.aargs, bargs

	def c_process_output(self, rs, iK):

		if not self.net.training:
			self.net.iK, self.net.real_iK, self.net.real_iV = iK, rs["net.real_iK"], rs["net.real_iV"]

		return rs["out"]

# Aggregation from: Exploiting Deep Representations for Neural Machine Translation
class ResidueCombiner(nn.Module):

	# isize: input size of Feed-forward NN

	def __init__(self, isize, ncomb=2, hsize=None, dropout=0.0, custom_act=use_adv_act_default, enable_bias=enable_prev_ln_bias_default, **kwargs):

		super(ResidueCombiner, self).__init__()

		_hsize = isize * 2 * ncomb if hsize is None else hsize

		# should dropout be in front of sigmoid or not?
		self.net = nn.Sequential(Linear(isize * ncomb, _hsize), Custom_Act() if custom_act else nn.Sigmoid(), Dropout(dropout, inplace=inplace_after_Custom_Act), Linear(_hsize, isize, bias=enable_bias), Dropout(dropout, inplace=True)) if dropout > 0.0 else nn.Sequential(Linear(isize * ncomb, _hsize), Custom_Act() if custom_act else nn.Sigmoid(), Linear(_hsize, isize, bias=enable_bias))

		self.out_normer = nn.LayerNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)

	def forward(self, *xl, **kwargs):

		# faster only when len(xl) is very large
		#out = torch.stack([self.net(torch.cat(xl, -1))] + list(xl), -2).sum(-2)
		out = self.net(torch.cat(xl, -1))
		for inputu in xl:
			out = out + inputu

		return self.out_normer(out)

class Scorer(nn.Module):

	def __init__(self, isize, bias=True, **kwargs):

		super(Scorer, self).__init__()

		self.w = nn.Parameter(torch.Tensor(isize).uniform_(- sqrt(1.0 / isize), sqrt(1.0 / isize)))
		self.bias = nn.Parameter(torch.zeros(1)) if bias else None

	def forward(self, x, **kwargs):

		xsize = x.size()

		out = torch.addmv(self.bias, x.view(-1, xsize[-1]), self.w) if self.bias else x.view(-1, xsize[-1]).mv(self.w)

		rsize = list(xsize)
		rsize[-1] = 1

		return out.view(rsize)

class NDWrapper(nn.Module):

	def __init__(self, module, num_dim, **kwargs):

		super(NDWrapper, self).__init__()

		self.net = module
		self.num_dim = num_dim

	def forward(self, x, *inputs, **kwargs):

		ndim = x.dim()
		if ndim == self.num_dim:
			return self.net(x, *inputs, **kwargs)
		else:
			return self.net(x.view(-1, *x.size()[1 - self.num_dim:]), *inputs, **kwargs).view(*x.size()[:-1], -1)

class GradientReversalFunction(Function):

	# Note that both forward and backward are @staticmethods
	@staticmethod
	def forward(ctx, inputs, adv_weight=1.0):

		ctx.adv_weight = adv_weight
		return inputs

	@staticmethod
	def backward(ctx, grad_outputs):

		if (grad_outputs is not None) and ctx.needs_input_grad[0]:
			_adv_weight = ctx.adv_weight
			return -grad_outputs if _adv_weight == 1.0 else (grad_outputs * -_adv_weight), None
		else:
			return None, None

GradientReversalFunc = GradientReversalFunction.apply

class GradientReversalLayer(nn.Module):

	def __init__(self, adv_weight=1.0, **kwargs):

		super(GradientReversalLayer, self).__init__()

		self.adv_weight = adv_weight

	def forward(self, *inputs, **kwargs):

		return tuple(GradientReversalFunc(inputu, self.adv_weight) for inputu in inputs) if len(inputs) > 1 else GradientReversalFunc(inputs[0], self.adv_weight)

class ACTLossFunction(Function):

	# Note that both forward and backward are @staticmethods
	@staticmethod
	def forward(ctx, weight, weight_loss, remain_value):

		ctx.save_for_backward(weight_loss, remain_value)

		return remain_value.sum()

	@staticmethod
	def backward(ctx, grad_output):

		weight_loss, remain_value = ctx.saved_tensors

		grad_weight = grad_output * weight_loss if ctx.needs_input_grad[0] else None

		grad_remain = grad_output.view(1, 1, 1).expand_as(remain_value) if ctx.needs_input_grad[2] else None

		return grad_weight, None, grad_remain

class ACT_Loss(nn.Module):

	def __init__(self):

		super(ACT_Loss, self).__init__()

	def forward(self, weight, weight_loss, remain_value, **kwargs):

		return ACTLossFunction.apply(weight, weight_loss, remain_value)

class ApproximateEmb(nn.Module):

	def __init__(self, weight, **kwargs):

		super(ApproximateEmb, self).__init__()
		self.weight = weight

	def forward(self, inpute, **kwargs):

		isize = list(inpute.size())
		out = inpute.view(-1, isize[-1])
		out = out.mm(self.weight)
		isize[-1] = -1
		return out.view(isize)

# SparseNormer is proposed in GLoMo: Unsupervisedly Learned Relational Graphs as Transferable Representations(https://arxiv.org/abs/1806.05662)
class SparseNormer(nn.Module):

	# dim: dimension to normalize

	def __init__(self, dim=-1, eps=ieps_default, **kwargs):

		super(SparseNormer, self).__init__()

		self.dim = dim
		self.bias = nn.Parameter(torch.zeros(1))
		self.act = nn.ReLU(inplace=True)
		self.eps = eps

	def forward(self, x, **kwargs):

		_tmp = self.act(x + self.bias)
		_tmp = _tmp * _tmp

		# fix zero-devision in case all elements in _tmp are 0.
		return _tmp / (_tmp.sum(self.dim, keepdim=True) + self.eps)

class MHSparseNormer(nn.Module):

	# nheads: number of heads
	# dim: dimension to normalize

	def __init__(self, nheads, dim=-1, eps=ieps_default, **kwargs):

		super(MHSparseNormer, self).__init__()

		self.dim = dim
		self.bias = nn.Parameter(torch.zeros(1, nheads, 1, 1))
		self.act = nn.ReLU(inplace=True)
		self.eps = eps

	# input should be: (bsize, nheads, nquery, seql)
	def forward(self, x, **kwargs):

		_tmp = self.act(x + self.bias)
		_tmp = _tmp * _tmp

		# fix zero-devision in case all elements in _tmp are 0.
		return _tmp / (_tmp.sum(self.dim, keepdim=True) + self.eps)

	def fix_init(self):

		with torch_no_grad():
			self.bias.data.zero_()

class MHAttnSummer(nn.Module):

	def __init__(self, isize, ahsize=None, num_head=8, attn_drop=0.0, **kwargs):

		super(MHAttnSummer, self).__init__()

		self.w = nn.Parameter(torch.Tensor(1, 1, isize).uniform_(- sqrt(1.0 / isize), sqrt(1.0 / isize)))
		self.attn = CrossAttn(isize, isize if ahsize is None else ahsize, isize, num_head, dropout=attn_drop)

	# x: (bsize, seql, isize)
	def forward(self, x, **kwargs):

		return self.attn(self.w, x).squeeze(1)

class FertSummer(nn.Module):

	def __init__(self, isize, **kwargs):

		super(FertSummer, self).__init__()

		self.net = Scorer(isize, False)
		self.normer = nn.Softmax(dim=1)

	# x: (bsize, seql, isize)
	def forward(self, x, mask=None, **kwargs):

		_weight = self.net(x)
		if mask is not None:
			_weight.masked_fill_(mask, -inf_default)

		# (bsize, seql, 1)" * (bsize, seql, isize) => (bsize, 1, isize)
		return self.normer(_weight).transpose(1, 2).bmm(x).squeeze(1)

class CoordinateEmb(nn.Module):

	# num_dim: dimension of embedding
	# num_pos: maximum length of sentence cached, extended length will be generated while needed and droped immediately after that
	# num_steps: similar to num_pos, but for steps
	# pos_offset: initial offset for position
	# dim_offset: initial offset for dimension

	def __init__(self, num_dim, num_pos=cache_len_default, num_steps=8, pos_offset=0, dim_offset=0, alpha=1.0, **kwargs):

		super(CoordinateEmb, self).__init__()

		self.num_pos = num_pos
		self.num_steps = num_steps
		self.num_dim = num_dim
		self.poff = pos_offset
		self.doff = dim_offset
		self.alpha = alpha
		self.register_buffer("w", torch.Tensor(num_steps, num_pos, num_dim), persistent=False)
		self.reset_parameters()

	# x: input (bsize, seql)

	def forward(self, x, step, expand=True, **kwargs):

		bsize, seql = x.size()[:2]

		if step <= self.num_steps:
			rs = self.w[step][:seql] if seql <= self.num_pos else torch.cat((self.w[step], self.get_ext(seql, step, False)), 0)
		else:
			rs = self.get_ext(seql, step, False)

		return rs.unsqueeze(0).expand(bsize, seql, self.num_dim) if expand else rs.unsqueeze(0)

	def reset_parameters(self):

		poff = self.poff
		npos = self.num_pos
		nstep = self.num_steps
		pos = torch.arange(poff, npos + poff, dtype=self.w.dtype, device=self.w.device).view(1, npos, 1)
		step = torch.arange(poff, nstep + poff, dtype=self.w.dtype, device=self.w.device).view(nstep, 1, 1)
		rdiv_term = (torch.arange(self.doff, self.num_dim + self.doff, 2, dtype=self.w.dtype, device=self.w.device) * -(log(1e4) / self.num_dim)).exp()
		_tmp1, _tmp2 = pos * rdiv_term, step * rdiv_term
		if self.alpha != 1.0:
			_tmp1.mul_(self.alpha)
			_tmp2.mul_(self.alpha)
		self.w[:, :, 0::2], self.w[:, :, 1::2] = _tmp1.sin() + _tmp2.sin(), ((_tmp1.cos() + _tmp2.cos()).narrow(-1, 0, _tmp1.size(-1) - 1) if self.num_dim % 2 == 1 else _tmp1.cos() + _tmp2.cos())

	def get_ext(self, length, step, step_pick=False):

		poff = self.poff
		_step = torch.as_tensor([step + poff], dtype=self.w.dtype, device=self.w.device).view(1, 1)

		if step_pick:
			_pos = torch.as_tensor([length + poff], dtype=self.w.dtype, device=self.w.device).view(1, 1)
			ed = self.w.new_empty(1, self.num_dim)
		else:
			npos = self.num_pos
			_pos = torch.arange(npos + poff if step <= self.num_steps else poff, length + poff, dtype=self.w.dtype, device=self.w.device).unsqueeze(1)
			ed = self.w.new_empty(length - npos, self.num_dim)
		rdiv_term = (torch.arange(self.doff, self.num_dim + self.doff, 2, dtype=self.w.dtype, device=self.w.device) * -(log(1e4) / self.num_dim)).exp()
		_tmp1, _tmp2 = _pos * rdiv_term, _step * rdiv_term
		if self.alpha != 1.0:
			_tmp1.mul_(self.alpha)
			_tmp2.mul_(self.alpha)
		ed[:, 0::2], ed[:, 1::2] = _tmp1.sin() + _tmp2.sin(), ((_tmp1.narrow(-1, 0, _tmp1.size(-1) - 1).cos() + _tmp2.narrow(-1, 0, _tmp1.size(-1) - 1).cos()) if self.num_dim % 2 == 1 else _tmp1.cos() + _tmp2.cos())

		return ed

	def get_pos(self, step, layer):

		return self.w[layer][step] if step < self.num_pos and layer < self.num_steps else self.get_ext(step, layer, True).squeeze(0)

class Temperature(nn.Module):

	def __init__(self, isize, minv=0.125, **kwargs):

		super(Temperature, self).__init__()

		self.w = nn.Parameter(torch.Tensor(isize).uniform_(- sqrt(1.0 / isize), sqrt(1.0 / isize)))
		self.bias = nn.Parameter(torch.zeros(1))
		self.act = nn.Tanh()
		self.k = nn.Parameter(torch.ones(1))
		self.minv = minv

	def forward(self, x, **kwargs):

		xsize = x.size()

		out = torch.addmv(self.bias, x.view(-1, xsize[-1]), self.w)

		xsize = list(xsize)
		xsize[-1] = 1

		return ((self.k.abs() + self.minv) * (self.act(out) + 1)).view(xsize)

	def fix_init(self):

		with torch_no_grad():
			self.k.data.fill_(1.0)
			self.bias.data.zero_()

def reduce_model(modin):

	rsm = reduce_model_list(modin, [PositionalEmb, CoordinateEmb], [lambda m: (m.num_pos, m.num_dim, m.poff, m.doff, m.alpha,), lambda m: (m.num_pos, m.num_dim, m.poff, m.doff, m.alpha, m.num_steps,),])

	return reduce_model_drop(reduce_model_act(rsm))
