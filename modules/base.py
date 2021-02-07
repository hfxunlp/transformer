#encoding: utf-8

from math import sqrt, log, exp
import torch
from torch import nn
from torch.nn import functional as nnFunc
from torch.autograd import Function

from utils.base import reduce_model_list
from modules.act import Custom_Act
from modules.act import reduce_model as reduce_model_act
from modules.dropout import Dropout
from modules.dropout import reduce_model as reduce_model_drop

from cnfg.ihyp import *

Linear = nn.Linear

class PositionwiseFF(nn.Module):

	# isize: input dimension
	# hsize: hidden dimension

	def __init__(self, isize, hsize=None, dropout=0.0, norm_residual=norm_residual_default, custom_act=use_adv_act_default, enable_bias=enable_prev_ln_bias_default):

		super(PositionwiseFF, self).__init__()

		_hsize = isize * 4 if hsize is None else hsize

		self.net = nn.Sequential(Linear(isize, _hsize), Custom_Act() if custom_act else nn.ReLU(inplace=True), Dropout(dropout, inplace=inplace_after_Custom_Act), Linear(_hsize, isize, bias=enable_bias), Dropout(dropout, inplace=True)) if dropout > 0.0 else nn.Sequential(Linear(isize, _hsize), Custom_Act() if custom_act else nn.ReLU(inplace=True), Linear(_hsize, isize, bias=enable_bias))

		self.normer = nn.LayerNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)

		self.norm_residual = norm_residual

	def forward(self, x):

		_out = self.normer(x)

		out = self.net(_out)

		out = out + (_out if self.norm_residual else x)

		return out

class PositionalEmb(nn.Module):

	# num_dim: dimension of embedding
	# num_pos: maximum length of sentence cached, extended length will be generated while needed and droped immediately after that
	# pos_offset: initial offset for position
	# dim_offset: initial offset for dimension

	def __init__(self, num_dim, num_pos=cache_len_default, pos_offset=0, dim_offset=0, alpha=1.0):

		super(PositionalEmb, self).__init__()

		self.num_pos = num_pos
		self.num_dim = num_dim
		self.poff = pos_offset
		self.doff = dim_offset
		self.alpha = alpha
		self.register_buffer('w', torch.Tensor(num_pos, num_dim))
		self.reset_parameters()

	# x: input (bsize, seql)

	def forward(self, x, expand=True):

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
			pos = torch.tensor([length + poff], dtype=self.w.dtype, device=self.w.device).unsqueeze(1)
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
	# sparsenorm: using sparse normer or standard softmax
	# bind_qk: query and key can share a same linear transformation for the Reformer: The Efficient Transformer (https://arxiv.org/abs/2001.04451) paper.

	def __init__(self, isize, hsize, osize, num_head=8, dropout=0.0, k_isize=None, v_isize=None, enable_bias=enable_prev_ln_bias_default, enable_proj_bias=enable_proj_bias_default, k_rel_pos=0, sparsenorm=False, bind_qk=False, xseql=cache_len_default):

		super(MultiHeadAttn, self).__init__()

		self.attn_dim = hsize // num_head
		self.hsize = self.attn_dim * num_head
		self.num_head = num_head

		self.query_adaptor = Linear(isize, self.hsize, bias=enable_proj_bias)
		_k_isize = isize if k_isize is None else k_isize
		self.key_adaptor = self.query_adaptor if bind_qk and isize == _k_isize else Linear(_k_isize, self.hsize, bias=enable_proj_bias)
		self.value_adaptor = Linear(_k_isize if v_isize is None else v_isize, self.hsize, bias=enable_proj_bias)

		self.outer = Linear(self.hsize, osize, bias=enable_bias)

		#self.normer = MHSparseNormer(num_head, dim=-1) if sparsenorm else nn.Softmax(dim=-1)
		self.normer = SparseNormer(dim=-1) if sparsenorm else nn.Softmax(dim=-1)

		self.drop = Dropout(dropout, inplace=sparsenorm) if dropout > 0.0 else None

		if k_rel_pos > 0:
			self.k_rel_pos = k_rel_pos
			self.rel_pemb = nn.Embedding(k_rel_pos * 2 + 1, self.attn_dim)
			_rpm = torch.arange(-xseql + 1, 1, dtype=torch.long).unsqueeze(0)
			self.register_buffer("rel_pos", (_rpm - _rpm.t()).clamp(min=-k_rel_pos, max=k_rel_pos) + k_rel_pos)
			self.xseql = xseql
			# the buffer can be shared inside the encoder or the decoder across layers for saving memory, by setting self.ref_rel_posm of self attns in deep layers to SelfAttn in layer 0, and sharing corresponding self.rel_pos
			self.ref_rel_posm = None
		else:
			self.rel_pemb = None

	# iQ: query (bsize, num_query, vsize)
	# iK: keys (bsize, seql, vsize)
	# iV: values (bsize, seql, vsize)
	# mask (bsize, num_query, seql)

	def forward(self, iQ, iK, iV, mask=None):

		bsize, nquery = iQ.size()[:2]
		seql = iK.size(1)
		nheads = self.num_head
		adim = self.attn_dim

		# real_iQ: MultiHead iQ (bsize, num_query, vsize) => (bsize, nheads, nquery, adim)
		# real_iK: MultiHead iK (bsize, seql, vsize) => (bsize, nheads, adim, seql)
		# real_iV: MultiHead iV (bsize, seql, vsize) => (bsize, nheads, seql, adim)

		real_iQ, real_iK, real_iV = self.query_adaptor(iQ).view(bsize, nquery, nheads, adim).transpose(1, 2), self.key_adaptor(iK).view(bsize, seql, nheads, adim).permute(0, 2, 3, 1), self.value_adaptor(iV).view(bsize, seql, nheads, adim).transpose(1, 2)

		# scores (bsize, nheads, nquery, adim) * (bsize, nheads, adim, seql) => (bsize, nheads, nquery, seql)

		scores = real_iQ.matmul(real_iK)

		if self.rel_pemb is not None:
			self.rel_pos_cache = self.get_rel_pos(seql).narrow(0, seql - nquery, nquery).contiguous() if self.ref_rel_posm is None else self.ref_rel_posm.rel_pos_cache
			scores += real_iQ.permute(2, 0, 1, 3).contiguous().view(nquery, bsize * nheads, adim).bmm(self.rel_pemb(self.get_rel_pos(seql).narrow(0, seql - nquery, nquery)).transpose(1, 2)).view(nquery, bsize, nheads, seql).permute(1, 2, 0, 3)

		scores = scores / sqrt(adim)

		if mask is not None:
			scores.masked_fill_(mask.unsqueeze(1), -inf_default)

		scores = self.normer(scores)

		if self.drop is not None:
			scores = self.drop(scores)

		# oMA: output of MultiHeadAttention T((bsize, nheads, nquery, seql) * (bsize, nheads, seql, adim)) => (bsize, nquery, nheads, adim)

		oMA = scores.matmul(real_iV).transpose(1, 2).contiguous()

		# output of this layer (bsize, nquery, nheads, adim) => (bsize, nquery, osize)

		return self.outer(oMA.view(bsize, nquery, self.hsize))

	def get_rel_pos(self, length):

		if length <= self.xseql:
			return self.rel_pos.narrow(0, 0, length).narrow(1, 0, length)
		else:
			_rpm = torch.arange(-length + 1, 1, dtype=self.rel_pos.dtype, device=self.rel_pos.device).unsqueeze(0)
			return ((_rpm - _rpm.t()).clamp(min=-self.k_rel_pos, max=self.k_rel_pos) + self.k_rel_pos)

# Average Attention is proposed in Accelerating Neural Transformer via an Average Attention Network (https://www.aclweb.org/anthology/P18-1166/)
class AverageAttn(nn.Module):

	# isize: input size of Feed-forward NN
	# hsize: hidden size of Feed-forward NN
	# dropout: dropout rate for Feed-forward NN
	# num_pos: maximum length of sentence cached, extended length will be generated while needed and droped immediately after that

	def __init__(self, isize, hsize=None, dropout=0.0, num_pos=cache_len_default, custom_act=use_adv_act_default):

		super(AverageAttn, self).__init__()

		_hsize = isize if hsize is None else hsize

		self.num_pos = num_pos
		self.register_buffer('w', torch.Tensor(num_pos, 1))

		self.ffn = nn.Sequential(Linear(isize, _hsize), Custom_Act() if custom_act else nn.ReLU(inplace=True), Dropout(dropout, inplace=inplace_after_Custom_Act), Linear(_hsize, isize), Dropout(dropout, inplace=True)) if dropout > 0.0 else nn.Sequential(Linear(isize, _hsize), Custom_Act() if custom_act else nn.ReLU(inplace=True), Linear(_hsize, isize))

		self.gw = Linear(isize * 2, isize * 2)

		self.reset_parameters()

	# iQ: keys (bsize, seql, vsize) for training, (bsize, 1, vsize) for decoding
	# iV: values (bsize, seql, vsize)
	# decoding: training state or decoding state

	def forward(self, iQ, iV, decoding=False):

		if decoding:
			avg = iV
		else:
			seql = iV.size(1)

			# avg: (bsize, seql, vsize)
			avg = iV.cumsum(dim=1) * (self.get_ext(seql) if seql > self.num_pos else self.w.narrow(0, 0, seql))

		avg = self.ffn(avg)

		igate, fgate = self.gw(torch.cat((iQ, avg), -1)).sigmoid().chunk(2, -1)

		return igate * iQ + fgate * avg

	def reset_parameters(self):

		self.w = self.get_ext(self.num_pos)

	def get_ext(self, npos):

		return (torch.arange(1, npos + 1, dtype=self.w.dtype, device=self.w.device).reciprocal_()).unsqueeze(-1)

# Accelerated MultiHeadAttn for self attention, use when Q == K == V
class SelfAttn(nn.Module):

	def __init__(self, isize, hsize, osize, num_head=8, dropout=0.0, enable_bias=enable_prev_ln_bias_default, enable_proj_bias=enable_proj_bias_default, k_rel_pos=use_k_relative_position, sparsenorm=False, xseql=cache_len_default):

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
			self.k_rel_pos = k_rel_pos
			self.rel_pemb = nn.Embedding(k_rel_pos * 2 + 1, self.attn_dim)
			_rpm = torch.arange(-xseql + 1, 1, dtype=torch.long).unsqueeze(0)
			self.register_buffer("rel_pos", (_rpm - _rpm.t()).clamp(min=-k_rel_pos, max=k_rel_pos) + k_rel_pos)
			self.xseql = xseql
			# the buffer can be shared inside the encoder or the decoder across layers for saving memory, by setting self.ref_rel_posm of self attns in deep layers to SelfAttn in layer 0, and sharing corresponding self.rel_pos
			self.ref_rel_posm = None
			self.register_buffer("rel_pos_cache", None)
		else:
			self.rel_pemb = None

	def forward(self, iQ, mask=None, iK=None):

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

		scores = self.normer(scores)

		if self.drop is not None:
			scores = self.drop(scores)

		oMA = scores.matmul(real_iV).transpose(1, 2).contiguous()

		return self.outer(oMA.view(bsize, nquery, self.hsize))

	def get_rel_pos(self, length):

		if length <= self.xseql:
			return self.rel_pos.narrow(0, 0, length).narrow(1, 0, length)
		else:
			_rpm = torch.arange(-length + 1, 1, dtype=self.rel_pos.dtype, device=self.rel_pos.device).unsqueeze(0)
			return ((_rpm - _rpm.t()).clamp(min=-self.k_rel_pos, max=self.k_rel_pos) + self.k_rel_pos)

# Accelerated MultiHeadAttn for cross attention, use when K == V
class CrossAttn(nn.Module):

	def __init__(self, isize, hsize, osize, num_head=8, dropout=0.0, k_isize=None, enable_bias=enable_prev_ln_bias_default, enable_proj_bias=enable_proj_bias_default, sparsenorm=False):

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

	def forward(self, iQ, iK, mask=None):

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

		scores = self.normer(scores)

		if self.drop is not None:
			scores = self.drop(scores)

		oMA = scores.matmul(real_iV).transpose(1, 2).contiguous()

		return self.outer(oMA.view(bsize, nquery, self.hsize))

# Aggregation from: Exploiting Deep Representations for Neural Machine Translation
class ResidueCombiner(nn.Module):

	# isize: input size of Feed-forward NN

	def __init__(self, isize, ncomb=2, hsize=None, dropout=0.0, custom_act=use_adv_act_default, enable_bias=enable_prev_ln_bias_default):

		super(ResidueCombiner, self).__init__()

		_hsize = isize * 2 * ncomb if hsize is None else hsize

		# should dropout be in front of sigmoid or not?
		self.net = nn.Sequential(Linear(isize * ncomb, _hsize), Custom_Act() if custom_act else nn.Sigmoid(), Dropout(dropout, inplace=inplace_after_Custom_Act), Linear(_hsize, isize, bias=enable_bias), Dropout(dropout, inplace=True)) if dropout > 0.0 else nn.Sequential(Linear(isize * ncomb, _hsize), Custom_Act() if custom_act else nn.Sigmoid(), Linear(_hsize, isize, bias=enable_bias))

		self.out_normer = nn.LayerNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)

	def forward(self, *xl):

		# faster only when len(xl) is very large
		#out = torch.stack([self.net(torch.cat(xl, -1))] + list(xl), -2).sum(-2)
		out = self.net(torch.cat(xl, -1))
		for inputu in xl:
			out = out + inputu

		return self.out_normer(out)

class Scorer(nn.Module):

	def __init__(self, isize, bias=True):

		super(Scorer, self).__init__()

		self.w = nn.Parameter(torch.Tensor(isize).uniform_(- sqrt(1.0 / isize), sqrt(1.0 / isize)))
		self.bias = nn.Parameter(torch.zeros(1)) if bias else None

	def forward(self, x):

		xsize = x.size()

		out = torch.addmv(self.bias, x.view(-1, xsize[-1]), self.w) if self.bias else x.view(-1, xsize[-1]).mv(self.w)

		rsize = list(xsize)
		rsize[-1] = 1

		return out.view(rsize)

class GradientReversalFunction(Function):

	# Note that both forward and backward are @staticmethods
	@staticmethod
	def forward(ctx, inputs, adv_weight=1.0):

		ctx.adv_weight = adv_weight
		return inputs

	@staticmethod
	def backward(ctx, grad_outputs):

		if grad_outputs is not None and ctx.needs_input_grad[0]:
			_adv_weight = ctx.adv_weight
			return -grad_outputs if _adv_weight == 1.0 else (grad_outputs * -_adv_weight), None
		else:
			return None, None

class GradientReversalLayer(nn.Module):

	def __init__(self, adv_weight=1.0):

		super(GradientReversalLayer, self).__init__()

		self.adv_weight = adv_weight

	def forward(self, *inputs):

		return (GradientReversalFunction.apply(inputu, self.adv_weight) for inputu in inputs) if len(inputs) > 1 else GradientReversalFunction.apply(inputs[0], self.adv_weight)

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

	def forward(self, weight, weight_loss, remain_value):

		return ACTLossFunction.apply(weight, weight_loss, remain_value)

class ApproximateEmb(nn.Module):

	def __init__(self, weight):

		super(ApproximateEmb, self).__init__()
		self.weight = weight

	def forward(self, inpute):

		isize = list(inpute.size())
		out = inpute.view(-1, isize[-1])
		out = out.mm(self.weight)
		isize[-1] = -1
		return out.view(isize)

# SparseNormer is proposed in GLoMo: Unsupervisedly Learned Relational Graphs as Transferable Representations(https://arxiv.org/abs/1806.05662)
class SparseNormer(nn.Module):

	# dim: dimension to normalize

	def __init__(self, dim=-1, eps=ieps_default):

		super(SparseNormer, self).__init__()

		self.dim = dim
		self.bias = nn.Parameter(torch.zeros(1))
		self.act = nn.ReLU(inplace=True)
		self.eps = eps

	def forward(self, x):

		_tmp = self.act(x + self.bias)
		_tmp = _tmp * _tmp

		# fix zero-devision in case all elements in _tmp are 0.
		return _tmp / (_tmp.sum(self.dim, keepdim=True) + self.eps)

class MHSparseNormer(nn.Module):

	# nheads: number of heads
	# dim: dimension to normalize

	def __init__(self, nheads, dim=-1, eps=ieps_default):

		super(MHSparseNormer, self).__init__()

		self.dim = dim
		self.bias = nn.Parameter(torch.zeros(1, nheads, 1, 1))
		self.act = nn.ReLU(inplace=True)
		self.eps = eps

	# input should be: (bsize, nheads, nquery, seql)
	def forward(self, x):

		_tmp = self.act(x + self.bias)
		_tmp = _tmp * _tmp

		# fix zero-devision in case all elements in _tmp are 0.
		return _tmp / (_tmp.sum(self.dim, keepdim=True) + self.eps)

	def fix_init(self):

		with torch.no_grad():
			self.bias.data.zero_()

class MHAttnSummer(nn.Module):

	def __init__(self, isize, ahsize=None, num_head=8, attn_drop=0.0):

		super(MHAttnSummer, self).__init__()

		self.w = nn.Parameter(torch.Tensor(1, 1, isize).uniform_(- sqrt(1.0 / isize), sqrt(1.0 / isize)))
		self.attn = CrossAttn(isize, isize if ahsize is None else ahsize, isize, num_head, dropout=attn_drop)

	# x: (bsize, seql, isize)
	def forward(self, x):

		return self.attn(self.w, x).squeeze(1)

class FertSummer(nn.Module):

	def __init__(self, isize):

		super(FertSummer, self).__init__()

		self.net = Scorer(isize, False)
		self.normer = nn.Softmax(dim=1)

	# x: (bsize, seql, isize)
	def forward(self, x, mask=None):

		_weight = self.net(x)
		if mask is not None:
			_weight.masked_fill_(mask, -inf_default)

		# (bsize, seql, 1)' * (bsize, seql, isize) => (bsize, 1, isize)
		return self.normer(_weight).transpose(1, 2).bmm(x).squeeze(1)

class CoordinateEmb(nn.Module):

	# num_dim: dimension of embedding
	# num_pos: maximum length of sentence cached, extended length will be generated while needed and droped immediately after that
	# num_steps: similar to num_pos, but for steps
	# pos_offset: initial offset for position
	# dim_offset: initial offset for dimension

	def __init__(self, num_dim, num_pos=cache_len_default, num_steps=8, pos_offset=0, dim_offset=0, alpha=1.0):

		super(CoordinateEmb, self).__init__()

		self.num_pos = num_pos
		self.num_steps = num_steps
		self.num_dim = num_dim
		self.poff = pos_offset
		self.doff = dim_offset
		self.alpha = alpha
		self.register_buffer('w', torch.Tensor(num_steps, num_pos, num_dim))
		self.reset_parameters()

	# x: input (bsize, seql)

	def forward(self, x, step, expand=True):

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
		_step = torch.tensor([step + poff], dtype=self.w.dtype, device=self.w.device).view(1, 1)

		if step_pick:
			_pos = torch.tensor([length + poff], dtype=self.w.dtype, device=self.w.device).view(1, 1)
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

	def __init__(self, isize, minv = 0.125):

		super(Temperature, self).__init__()

		self.w = nn.Parameter(torch.Tensor(isize).uniform_(- sqrt(1.0 / isize), sqrt(1.0 / isize)))
		self.bias = nn.Parameter(torch.zeros(1))
		self.act = nn.Tanh()
		self.k = nn.Parameter(torch.ones(1))
		self.minv = minv

	def forward(self, x):

		xsize = x.size()

		out = torch.addmv(self.bias, x.view(-1, xsize[-1]), self.w)

		xsize = list(xsize)
		xsize[-1] = 1

		return ((self.k.abs() + self.minv) * (self.act(out) + 1)).view(xsize)

	def fix_init(self):

		with torch.no_grad():
			self.k.data.fill_(1.0)
			self.bias.data.zero_()

def reduce_model(modin):

	rsm = reduce_model_list(modin, [PositionalEmb, CoordinateEmb], [lambda m: (m.num_pos, m.num_dim, m.poff, m.doff, m.alpha,), lambda m: (m.num_pos, m.num_dim, m.poff, m.doff, m.alpha, m.num_steps,),])

	return reduce_model_drop(reduce_model_act(rsm))
