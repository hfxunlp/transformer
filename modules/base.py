#encoding: utf-8

from math import sqrt, log, exp, pi
import torch
from torch import nn
from torch.nn import functional as nnFunc
from torch.autograd import Function

class PositionwiseFF(nn.Module):

	# isize: input dimension
	# hsize: hidden dimension

	def __init__(self, isize, hsize=None, dropout=0.0, norm_residue=False, use_GeLU=False):

		super(PositionwiseFF, self).__init__()

		_hsize = isize * 4 if hsize is None else hsize

		if dropout > 0.0:
			self.nets = nn.ModuleList([nn.Linear(isize, _hsize), nn.Dropout(dropout, inplace=True), GeLU_BERT() if use_GeLU else nn.ReLU(inplace=True), nn.Linear(_hsize, isize), nn.Dropout(dropout, inplace=True)])
		else:
			self.nets = nn.ModuleList([nn.Linear(isize, _hsize), GeLU_BERT() if use_GeLU else nn.ReLU(inplace=True), nn.Linear(_hsize, isize)])

		self.normer = nn.LayerNorm(isize, eps=1e-06)

		self.norm_residue = norm_residue

	def forward(self, x):

		_out = self.normer(x)

		out = _out
		for net in self.nets:
			out = net(out)

		out = out + (_out if self.norm_residue else x)

		return out

class PositionalEmb(nn.Module):

	# num_dim: dimension of embedding
	# num_pos: maximum length of sentence cached, extended length will be generated while needed and droped immediately after that
	# pos_offset: initial offset for position
	# dim_offset: initial offset for dimension

	def __init__(self, num_dim, num_pos=512, pos_offset=0, dim_offset=0):

		super(PositionalEmb, self).__init__()

		self.num_pos = num_pos
		self.num_dim = num_dim
		self.poff = pos_offset
		self.doff = dim_offset
		self.register_buffer('w', torch.Tensor(num_pos, num_dim))
		self.reset_parameters()

	# x: input (bsize, seql)

	def forward(self, x, expand=True):

		bsize, seql = x.size()

		rs = self.w[:seql].unsqueeze(0) if seql <= self.num_pos else torch.cat((self.w, self.get_ext(seql, False)), 0).unsqueeze(0)

		return rs.expand(bsize, seql, self.num_dim) if expand else rs

	# when self.num_dim % 2 == 1, a bug happened, since rdiv_term for sin and cos are different

	def reset_parameters(self):

		poff = self.poff
		pos = torch.arange(poff, self.num_pos + poff, dtype=self.w.dtype, device=self.w.device).unsqueeze(1)
		rdiv_term = (torch.arange(self.doff, self.num_dim + self.doff, 2, dtype=self.w.dtype, device=self.w.device) * -(log(10000.0) / self.num_dim)).exp()
		_tmp = pos * rdiv_term
		self.w[:, 0::2], self.w[:, 1::2] = _tmp.sin(), _tmp.cos()

	def get_ext(self, length, step_pick=False):

		poff = self.poff

		if step_pick:
			pos = torch.Tensor([length + poff], dtype=self.w.dtype, device=self.w.device).unsqueeze(1)
			ed = self.w.new(1, self.num_dim)
		else:
			npos = self.num_pos
			pos = torch.arange(npos + poff, length + poff, dtype=self.w.dtype, device=self.w.device).unsqueeze(1)
			ed = self.w.new(length - npos, self.num_dim)
		rdiv_term = (torch.arange(self.doff, self.num_dim + self.doff, 2, dtype=self.w.dtype, device=self.w.device) * -(log(10000.0) / self.num_dim)).exp()
		_tmp = pos * rdiv_term
		ed[:, 0::2], ed[:, 1::2] = _tmp.sin(), _tmp.cos()

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

	def __init__(self, isize, hsize, osize, num_head=8, dropout=0.0, enable_bias=False, sparsenorm=False):

		super(MultiHeadAttn, self).__init__()

		self.attn_dim = hsize // num_head
		self.hsize = self.attn_dim * num_head
		self.num_head = num_head

		self.query_adaptor = nn.Linear(isize, self.hsize, bias=enable_bias)
		self.value_adaptor = nn.Linear(isize, self.hsize, bias=enable_bias)
		self.key_adaptor = nn.Linear(isize, self.hsize, bias=enable_bias)

		self.outer = nn.Linear(self.hsize, osize, bias=enable_bias)

		#self.normer = MHSparseNormer(num_head, dim=-1) if sparsenorm else nn.Softmax(dim=-1)
		self.normer = SparseNormer(dim=-1) if sparsenorm else nn.Softmax(dim=-1)

		self.drop = nn.Dropout(dropout, inplace=sparsenorm) if dropout > 0.0 else None

	# iQ: query (bsize, num_query, vsize)
	# iK: keys (bsize, seql, vsize)
	# iV: values (bsize, seql, vsize)
	# mask (bsize, num_query, seql)

	def forward(self, iQ, iK, iV, mask=None):

		bsize, nquery, _ = iQ.size()
		seql = iK.size(1)
		nheads = self.num_head
		adim = self.attn_dim

		# real_iQ: MultiHead iQ (bsize, num_query, vsize) => (bsize, nheads, nquery, adim)
		# real_iK: MultiHead iK (bsize, seql, vsize) => (bsize, nheads, adim, seql)
		# real_iV: MultiHead iV (bsize, seql, vsize) => (bsize, nheads, seql, adim)

		real_iQ, real_iK, real_iV = self.query_adaptor(iQ).view(bsize, nquery, nheads, adim).transpose(1, 2), self.key_adaptor(iK).view(bsize, seql, nheads, adim).permute(0, 2, 3, 1), self.value_adaptor(iV).view(bsize, seql, nheads, adim).transpose(1, 2)

		# scores (bsize, nheads, nquery, adim) * (bsize, nheads, adim, seql) => (bsize, nheads, nquery, seql)

		scores = real_iQ.matmul(real_iK) / sqrt(adim)

		if mask is not None:
			scores.masked_fill_(mask.unsqueeze(1).expand_as(scores), -1e32)

		scores = self.normer(scores)

		if self.drop is not None:
			scores = self.drop(scores)

		# oMA: output of MultiHeadAttention T((bsize, nheads, nquery, seql) * (bsize, nheads, seql, adim)) => (bsize, nquery, nheads, adim)

		oMA = scores.matmul(real_iV).transpose(1, 2).contiguous()

		# output of this layer (bsize, nquery, nheads, adim) => (bsize, nquery, osize)

		return self.outer(oMA.view(bsize, nquery, self.hsize))

# Average Attention is proposed in Accelerating Neural Transformer via an Average Attention Network(https://arxiv.org/abs/1805.00631)

class AverageAttn(nn.Module):

	# isize: input size of Feed-forward NN
	# hsize: hidden size of Feed-forward NN
	# dropout: dropout rate for Feed-forward NN
	# num_pos: maximum length of sentence cached, extended length will be generated while needed and droped immediately after that

	def __init__(self, isize, hsize=None, dropout=0.0, num_pos=512, use_GeLU=False):

		super(AverageAttn, self).__init__()

		_hsize = isize if hsize is None else hsize

		self.num_pos = num_pos
		self.register_buffer('w', torch.Tensor(num_pos, num_pos))

		self.ffn = nn.Sequential(nn.Linear(isize, _hsize), nn.Dropout(dropout, inplace=True), GeLU_BERT() if use_GeLU else nn.ReLU(inplace=True), nn.Linear(_hsize, isize), nn.Dropout(dropout, inplace=True)) if dropout > 0.0 else nn.Sequential(nn.Linear(isize, _hsize), GeLU_BERT() if use_GeLU else nn.ReLU(inplace=True), nn.Linear(_hsize, isize))

		self.gw = nn.Linear(isize * 2, isize * 2)

		self.reset_parameters()

	# iQ: keys (bsize, seql, vsize) for training, (bsize, 1, vsize) for decoding
	# iV: values (bsize, seql, vsize)
	# decoding: training state or decoding state

	def forward(self, iQ, iV, decoding=False):

		if decoding:
			avg = iV
		else:
			bsize, seql, _ = iV.size()

			# attn: (seql, seql)
			if seql > self.num_pos:
				attn = self.get_ext(seql)
			else:
				attn = self.w.narrow(0, 0, seql).narrow(1, 0, seql)

			# avg: (bsize, seql, vsize)
			avg = attn.unsqueeze(0).expand(bsize, seql, seql).matmul(iV)

		avg = self.ffn(avg)

		ifg = self.gw(torch.cat((iQ, avg), -1)).sigmoid()
		isize = avg.size(-1)
		igate = ifg.narrow(-1, 0, isize)
		fgate = ifg.narrow(-1, isize, isize)

		return igate * iQ + fgate * avg

	def reset_parameters(self):

		self.w = self.get_ext(self.num_pos)

	def get_ext(self, npos):

		_tmp = (1.0 / torch.arange(1, npos + 1, dtype=self.w.dtype, device=self.w.device)).unsqueeze(1).repeat(1, npos)

		return _tmp.tril(0)

# Accelerated MultiHeadAttn for self attention, use when Q == K == V
class SelfAttn(nn.Module):

	# isize: input dimension
	# hsize: hidden dimension
	# osize: output size of this layer
	# num_head: number of heads
	# dropout: dropout probability
	# sparsenorm: using sparse normer or standard softmax

	def __init__(self, isize, hsize, osize, num_head=8, dropout=0.0, enable_bias=False, sparsenorm=False):

		super(SelfAttn, self).__init__()

		self.attn_dim = hsize // num_head
		self.hsize = self.attn_dim * num_head
		self.num_head = num_head

		self.adaptor = nn.Linear(isize, self.hsize * 3, bias=enable_bias)

		self.outer = nn.Linear(self.hsize, osize, bias=enable_bias)

		#self.normer = MHSparseNormer(num_head, dim=-1) if sparsenorm else nn.Softmax(dim=-1)
		self.normer = SparseNormer(dim=-1) if sparsenorm else nn.Softmax(dim=-1)

		self.drop = nn.Dropout(dropout, inplace=sparsenorm) if dropout > 0.0 else None

	# iQ: query (bsize, num_query, vsize)
	# mask (bsize, num_query, seql)
	# iK: key/value (bsize, seql, vsize), in case key != query, for efficient decoding

	def forward(self, iQ, mask=None, iK=None):

		bsize, nquery, _ = iQ.size()
		nheads = self.num_head
		adim = self.attn_dim

		# real_iQ: MultiHead iQ (bsize, num_query, vsize) => (bsize, nheads, nquery, adim)
		# real_iK: MultiHead iK (bsize, nquery, vsize) => (bsize, nheads, adim, seql)
		# real_iV: MultiHead iV (bsize, nquery, vsize) => (bsize, nheads, seql, adim)

		if iK is None:

			real_iQ, real_iK, real_iV = self.adaptor(iQ).view(bsize, nquery, 3, nheads, adim).unbind(2)

			real_iQ, real_iK, real_iV = real_iQ.transpose(1, 2), real_iK.permute(0, 2, 3, 1), real_iV.transpose(1, 2)
			
		else:

			seql = iK.size(1)

			real_iQ, _out = nnFunc.linear(iQ, self.adaptor.weight.narrow(0, 0, self.hsize), self.adaptor.bias.narrow(0, 0, self.hsize) if self.adaptor.bias else None).view(bsize, nquery, nheads, adim).transpose(1, 2), nnFunc.linear(iK, self.adaptor.weight.narrow(0, self.hsize, self.hsize + self.hsize), self.adaptor.bias.narrow(0, self.hsize, self.hsize + self.hsize) if self.adaptor.bias else None).view(bsize, seql, 2, nheads, adim)

			real_iK, real_iV = _out.unbind(2)
			real_iK, real_iV = real_iK.permute(0, 2, 3, 1), real_iV.transpose(1, 2)

		# scores (bsize, nheads, nquery, adim) * (bsize, nheads, adim, seql) => (bsize, nheads, nquery, seql)

		scores = real_iQ.matmul(real_iK) / sqrt(adim)

		if mask is not None:
			scores.masked_fill_(mask.unsqueeze(1).expand_as(scores), -1e32)

		scores = self.normer(scores)

		if self.drop is not None:
			scores = self.drop(scores)

		# oMA: output of MultiHeadAttention T((bsize, nheads, nquery, nquery) * (bsize, nheads, nquery, adim)) => (bsize, nquery, nheads, adim)

		oMA = scores.matmul(real_iV).transpose(1, 2).contiguous()

		# output of this layer (bsize, nquery, nheads, adim) => (bsize, nquery, osize)

		return self.outer(oMA.view(bsize, nquery, self.hsize))

# Accelerated MultiHeadAttn for cross attention, use when K == V
class CrossAttn(nn.Module):

	# isize: input dimension
	# hsize: hidden dimension
	# osize: output size of this layer
	# num_head: number of heads
	# dropout: dropout probability
	# sparsenorm: using sparse normer or standard softmax

	def __init__(self, isize, hsize, osize, num_head=8, dropout=0.0, enable_bias=False, sparsenorm=False):

		super(CrossAttn, self).__init__()

		self.attn_dim = hsize // num_head
		self.hsize = self.attn_dim * num_head
		self.num_head = num_head

		self.query_adaptor = nn.Linear(isize, self.hsize, bias=enable_bias)
		self.kv_adaptor = nn.Linear(isize, self.hsize * 2, bias=enable_bias)

		self.outer = nn.Linear(self.hsize, osize, bias=enable_bias)

		#self.normer = MHSparseNormer(num_head, dim=-1) if sparsenorm else nn.Softmax(dim=-1)
		self.normer = SparseNormer(dim=-1) if sparsenorm else nn.Softmax(dim=-1)

		self.drop = nn.Dropout(dropout, inplace=sparsenorm) if dropout > 0.0 else None

	# iQ: query (bsize, num_query, vsize)
	# iK: keys (bsize, seql, vsize)
	# mask (bsize, num_query, seql)

	def forward(self, iQ, iK, mask=None):

		bsize, nquery, _ = iQ.size()
		seql = iK.size(1)
		nheads = self.num_head
		adim = self.attn_dim

		# real_iQ: MultiHead iQ (bsize, num_query, vsize) => (bsize, nheads, nquery, adim)
		# real_iK: MultiHead iK (bsize, seql, vsize) => (bsize, nheads, seql, adim)
		# real_iV: MultiHead iV (bsize, seql, vsize) => (bsize, nheads, seql, adim)

		real_iQ, _out = self.query_adaptor(iQ).view(bsize, nquery, nheads, adim).transpose(1, 2), self.kv_adaptor(iK).view(bsize, seql, 2, nheads, adim)

		real_iK, real_iV = _out.unbind(2)
		real_iK, real_iV = real_iK.permute(0, 2, 3, 1), real_iV.transpose(1, 2)

		# scores (bsize, nheads, nquery, adim) * (bsize, nheads, adim, seql) => (bsize, nheads, nquery, seql)

		scores = real_iQ.matmul(real_iK) / sqrt(adim)

		if mask is not None:
			scores.masked_fill_(mask.unsqueeze(1).expand_as(scores), -1e32)

		scores = self.normer(scores)

		if self.drop is not None:
			scores = self.drop(scores)

		# oMA: output of MultiHeadAttention T((bsize, nheads, nquery, seql) * (bsize, nheads, seql, adim)) => (bsize, nquery, nheads, adim)

		oMA = scores.matmul(real_iV).transpose(1, 2).contiguous()

		# output of this layer (bsize, nquery, nheads, adim) => (bsize, nquery, osize)

		return self.outer(oMA.view(bsize, nquery, self.hsize))

# Aggregation from: Exploiting Deep Representations for Neural Machine Translation
class ResidueCombiner(nn.Module):

	# isize: input size of Feed-forward NN

	def __init__(self, isize, ncomb=2, hsize=None, use_GeLU=False):

		super(ResidueCombiner, self).__init__()

		_hsize = isize * 2 * ncomb if hsize is None else hsize

		self.net = nn.Sequential(nn.Linear(isize * ncomb, _hsize), GeLU_BERT() if use_GeLU else nn.Sigmoid(), nn.Linear(_hsize, isize))

		self.out_normer = nn.LayerNorm(isize, eps=1e-06)

	def forward(self, *xl):

		out = torch.stack([self.net(torch.cat(xl, -1))] + list(xl), -2).sum(-2)

		return self.out_normer(out)

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

class GradientReversalFunction(Function):

	# Note that both forward and backward are @staticmethods
	@staticmethod
	def forward(ctx, inputs):

		return inputs

	@staticmethod
	def backward(ctx, grad_outputs):

		return - grad_outputs if grad_outputs is not None and ctx.needs_input_grad[0] else None

class GradientReversalLayer(nn.Module):

	def __init__(self):

		super(GradientReversalLayer, self).__init__()

	def forward(self, *inputs):

		return (GradientReversalFunction.apply(inputu) for inputu in inputs) if len(inputs) > 1 else GradientReversalFunction.apply(inputs[0])


# SparseMax (https://arxiv.org/pdf/1602.02068) borrowed form OpenNMT-py( https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/sparse_activations.py)
class SparsemaxFunction(Function):

	@staticmethod
	def forward(ctx, input, dim=0):

		def _threshold_and_support(input, dim=0):

			def _make_ix_like(input, dim=0):

				d = input.size(dim)
				rho = torch.arange(1, d + 1, device=input.device, dtype=input.dtype)
				view = [1] * input.dim()
				view[0] = -1

				return rho.view(view).transpose(0, dim)

			input_srt, _ = input.sort(descending=True, dim=dim)
			input_cumsum = input_srt.cumsum(dim) - 1
			rhos = _make_ix_like(input, dim)
			support = rhos * input_srt > input_cumsum

			support_size = support.sum(dim=dim).unsqueeze(dim)
			tau = input_cumsum.gather(dim, support_size - 1)
			tau /= support_size.to(input.dtype)

			return tau, support_size

		ctx.dim = dim
		max_val, _ = input.max(dim=dim, keepdim=True)
		input -= max_val
		tau, supp_size = _threshold_and_support(input, dim=dim)
		output = (input - tau).clamp(min=0)
		ctx.save_for_backward(supp_size, output)

		return output

	@staticmethod
	def backward(ctx, grad_output):

		supp_size, output = ctx.saved_tensors
		dim = ctx.dim
		grad_input = grad_output.clone()
		grad_input[output == 0] = 0

		v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()
		v_hat = v_hat.unsqueeze(dim)
		grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)

		return grad_input, None

class Sparsemax(nn.Module):

	def __init__(self, dim=0):

		super(Sparsemax, self).__init__()
		self.dim = dim

	def forward(self, input):

		return SparsemaxFunction.apply(input, self.dim)

class SigmoidIncremental:

	# warm_steps: increase from 0.0 to about (0.9866 * target_value) in warm_steps
	# target_value: target value returned after infinity calls to step() function

	def __init__(self, warm_steps, target_value, cur_step=0):

		self.wstep = float(warm_steps) /  5.0
		self.tarv = target_value
		self.cur_step = float(cur_step)

	def step(self):

		self.cur_step += 1.0

		return (2.0 / (1.0 + exp(- self.cur_step / self.wstep)) - 1.0) * self.tarv

class SigmoidITensor(nn.Module):

	# warm_steps: increase from 0.0 to about (0.9866 * target_value) in warm_steps
	# target_value: target value returned after infinity calls to step() function

	def __init__(self, warm_steps, target_value, xseql=512):

		super(SigmoidITensor, self).__init__()
		self.wstep = float(warm_steps) /  5.0
		self.tarv = target_value
		self.xseql = xseql
		self.register_buffer("w", ((((torch.arange(1, xseql + 1, dtype=torch.float, requires_grad=False) / self.wstep)).sigmoid() * 2 - 1) * self.tarv).unsqueeze(0).unsqueeze(-1))

	def forward(self, x, expand=True):

		seql = x.size(1)

		out = self.get_ext(seql) if seql > self.xseql else self.w.narrow(1, 0, seql)

		return out.expand_as(x) if expand else out

	def get_ext(self, seql):

		_tmp = ((((torch.arange(self.xseql + 1, seql + 1, dtype=self.w.dtype, device=self.w.device, requires_grad=False) / self.wstep)).sigmoid() * 2.0 - 1.0) * self.tarv).unsqueeze(0).unsqueeze(-1)

		return torch.cat((self.w, _tmp), 1)

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

# 2 kinds of GELU activation function implementation according to https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py#L53-L58

class GeLU_GPT(nn.Module):

	def __init__(self):

		super(GeLU_GPT, self).__init__()

		self.k = sqrt(2.0 / pi)

	def forward(self, x):

		return 0.5 * x * (1.0 + (self.k * (x + 0.044715 * x.pow(3.0))).tanh())

class GeLU_BERT(nn.Module):

	def __init__(self):

		super(GeLU_BERT, self).__init__()

		self.k = sqrt(2.0)

	def forward(self, x):

		return 0.5 * x * (1.0 + (x / self.k).erf())

# SparseNormer is proposed in GLoMo: Unsupervisedly Learned Relational Graphs as Transferable Representations(https://arxiv.org/abs/1806.05662)

class SparseNormer(nn.Module):

	# dim: dimension to normalize

	def __init__(self, dim=-1, ieps=1e-32):

		super(SparseNormer, self).__init__()

		self.dim = dim
		self.bias = nn.Parameter(torch.zeros(1))
		self.act = nn.ReLU(inplace=True)
		self.ieps = ieps

	def forward(self, x):

		_tmp = self.act(x + self.bias)
		_tmp = _tmp * _tmp

		# fix zero-devision in case all elements in _tmp are 0.
		return _tmp / (_tmp.sum(self.dim, keepdim=True) + self.ieps)

class MHSparseNormer(nn.Module):

	# nheads: number of heads
	# dim: dimension to normalize

	def __init__(self, nheads, dim=-1, ieps=1e-32):

		super(MHSparseNormer, self).__init__()

		self.dim = dim
		self.bias = nn.Parameter(torch.zeros(1, nheads, 1, 1))
		self.act = nn.ReLU(inplace=True)
		self.ieps = ieps

	# input should be: (bsize, nheads, nquery, seql)
	def forward(self, x):

		_tmp = self.act(x + self.bias)
		_tmp = _tmp * _tmp

		# fix zero-devision in case all elements in _tmp are 0.
		return _tmp / (_tmp.sum(self.dim, keepdim=True) + self.ieps)

	def fix_init(self):

		self.bias.data.zero_()

class Scorer(nn.Module):

	def __init__(self, isize, bias=True):

		super(Scorer, self).__init__()

		self.w = nn.Parameter(torch.Tensor(isize).uniform_(- sqrt(6.0 / isize), sqrt(6.0 / isize)))
		self.bias = nn.Parameter(torch.zeros(1)) if bias else None

	def forward(self, x):

		xsize = x.size()

		out = torch.addmv(self.bias, x.view(-1, xsize[-1]), self.w) if self.bias else x.view(-1, xsize[-1]).mv(self.w)

		rsize = list(xsize)
		rsize[-1] = 1

		return out.view(rsize)

class MHAttnSummer(nn.Module):

	def __init__(self, isize, ahsize=None, num_head=8, attn_drop=0.0):

		super(MHAttnSummer, self).__init__()

		self.w = nn.Parameter(torch.Tensor(1, 1, isize).uniform_(- sqrt(6.0 / isize), sqrt(6.0 / isize)))
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
			_weight.masked_fill_(mask, -1e32)

		# (bsize, seql, 1)' * (bsize, seql, isize) => (bsize, 1, isize)
		return self.normer(_weight).transpose(1, 2).bmm(x).squeeze(1)

class Temperature(nn.Module):

	def __init__(self, isize, minv = 0.125):

		super(Temperature, self).__init__()

		self.w = nn.Parameter(torch.Tensor(isize).uniform_(- sqrt(6.0 / isize), sqrt(6.0 / isize)))
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

		self.k.data.fill_(1.0)
		self.bias.data.zero_()

class CoordinateEmb(nn.Module):

	# num_dim: dimension of embedding
	# num_pos: maximum length of sentence cached, extended length will be generated while needed and droped immediately after that
	# num_steps: similar to num_pos, but for steps
	# pos_offset: initial offset for position
	# dim_offset: initial offset for dimension

	def __init__(self, num_dim, num_pos=512, num_steps=8, pos_offset=0, dim_offset=0):

		super(CoordinateEmb, self).__init__()

		self.num_pos = num_pos
		self.num_steps = num_steps
		self.num_dim = num_dim
		self.poff = pos_offset
		self.doff = dim_offset
		self.register_buffer('w', torch.Tensor(num_steps, num_pos, num_dim))
		self.reset_parameters()

	# x: input (bsize, seql)

	def forward(self, x, step, expand=True):

		bsize, seql = x.size()[:2]

		if step < self.num_steps:
			rs = self.w[step][:seql] if seql <= self.num_pos else torch.cat((self.w[step], self.get_ext(seql, step, False)), 0)
		else:
			rs = self.get_ext(seql, step, False)

		return rs.unsqueeze(0).expand(bsize, seql, self.num_dim) if expand else rs.unsqueeze(0)

	# when self.num_dim % 2 == 1, a bug happened, since rdiv_term for sin and cos are different

	def reset_parameters(self):

		poff = self.poff
		npos = self.num_pos
		nstep = self.num_steps
		pos = torch.arange(poff, npos + poff, dtype=self.w.dtype, device=self.w.device).view(1, npos, 1)
		step = torch.arange(poff, nstep + poff, dtype=self.w.dtype, device=self.w.device).view(nstep, 1, 1)
		rdiv_term = (torch.arange(self.doff, self.num_dim + self.doff, 2, dtype=self.w.dtype, device=self.w.device) * -(log(10000.0) / self.num_dim)).exp()
		_tmp1, _tmp2 = pos * rdiv_term, step * rdiv_term
		self.w[:, :, 0::2], self.w[:, :, 1::2] = _tmp1.sin() + _tmp2.sin(), _tmp1.cos() + _tmp2.cos()

	def get_ext(self, length, step, step_pick=False):

		poff = self.poff
		_step = torch.Tensor([step + poff], dtype=self.w.dtype, device=self.w.device).view(1, 1)

		if step_pick:
			_pos = torch.Tensor([length + poff], dtype=self.w.dtype, device=self.w.device).view(1, 1)
			ed = self.w.new(1, self.num_dim)
		else:
			npos = self.num_pos
			_pos = torch.arange(npos + poff if step < self.num_steps else poff, length + poff, dtype=self.w.dtype, device=self.w.device).unsqueeze(1)
			ed = self.w.new(length - npos, self.num_dim)
		rdiv_term = (torch.arange(self.doff, self.num_dim + self.doff, 2, dtype=self.w.dtype, device=self.w.device) * -(log(10000.0) / self.num_dim)).exp()
		_tmp1, _tmp2 = _pos * rdiv_term, _step * rdiv_term
		ed[:, 0::2], ed[:, 1::2] = _tmp1.sin() + _tmp2.sin(), _tmp1.cos() + _tmp2.cos()

		return ed

	# step of weight to retrieve, start from 0

	def get_pos(self, step, layer):

		return self.w[layer][step] if step < self.num_pos and layer < self.num_steps else self.get_ext(step, layer, True).squeeze(0)

class GausNoiser(nn.Module):

	def __init__(self, power):

		super(GausNoiser, self).__init__()
		self.power = power

	# mask: (bsize, seql, 1), otherwise cannot multiply with inpute.size(-1)
	def forward(self, inpute, mask=None):

		if self.training:
			if mask is None:
				base_p = inpute.data.abs().mean() * self.power
			else:
				base_p = inpute.data.abs().masked_fill(mask, 0.0).sum() * (self.power / float((mask.numel() - mask.sum().item()) * inpute.size(-1)))

			return torch.randn(inpute.size(), dtype=inpute.dtype, device=inpute.device) * base_p + inpute

		return inpute

class UniNoiser(nn.Module):

	def __init__(self, power):

		super(UniNoiser, self).__init__()
		self.power = power

	# mask: (bsize, seql, 1), otherwise cannot multiply with inpute.size(-1)
	def forward(self, inpute, mask=None):

		if self.training:
			if mask is None:
				base_p = inpute.data.abs().mean().item() * self.power
			else:
				base_p = inpute.data.abs().masked_fill(mask, 0.0).sum().item() / float((mask.numel() - mask.sum().item()) * inpute.size(-1)) * self.power

			return inpute.new_empty(inpute.size(), requires_grad=False).uniform_(- base_p, base_p) + inpute

		return inpute
