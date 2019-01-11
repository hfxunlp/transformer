#encoding: utf-8

from math import sqrt, log, exp, pi
import torch
from torch import nn
from torch.autograd import Function

class PositionwiseFF(nn.Sequential):

	# isize: input dimension
	# hsize: hidden dimension

	def __init__(self, isize, hsize=None, dropout=0.0, norm_input=True, use_GeLU=False):

		_hsize = isize * 4 if hsize is None else hsize

		if norm_input:
			if dropout > 0.0:
				super(PositionwiseFF, self).__init__(nn.LayerNorm(isize, eps=1e-06), nn.Linear(isize, _hsize), nn.Dropout(dropout), GeLU_BERT() if use_GeLU else nn.ReLU(), nn.Linear(_hsize, isize), nn.Dropout(dropout))
			else:
				super(PositionwiseFF, self).__init__(nn.LayerNorm(isize, eps=1e-06), nn.Linear(isize, _hsize), GeLU_BERT() if use_GeLU else nn.ReLU(), nn.Linear(_hsize, isize))
		else:
			if dropout > 0.0:
				super(PositionwiseFF, self).__init__(nn.Linear(isize, _hsize), nn.Dropout(dropout), GeLU_BERT() if use_GeLU else nn.ReLU(), nn.Linear(_hsize, isize), nn.Dropout(dropout))
			else:
				super(PositionwiseFF, self).__init__(nn.Linear(isize, _hsize), GeLU_BERT() if use_GeLU else nn.ReLU(), nn.Linear(_hsize, isize))


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
		self.register_buffer('w', torch.Tensor(num_pos, num_dim))
		self.register_buffer("rdiv_term", torch.exp(torch.arange(dim_offset, num_dim + dim_offset, 2, dtype=self.w.dtype, device=self.w.device) * -(log(10000.0) / num_dim)))
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
		self.w[:, 0::2], self.w[:, 1::2] = torch.sin(pos * self.rdiv_term), torch.cos(pos * self.rdiv_term)

	def get_ext(self, length, step_pick=False):

		poff = self.poff

		if step_pick:
			pos = torch.Tensor([length + poff], dtype=self.w.dtype, device=self.w.device).unsqueeze(1)
			ed = self.w.new(1, self.num_dim)
		else:
			npos = self.num_pos
			pos = torch.arange(npos + poff, length + poff, dtype=self.w.dtype, device=self.w.device).unsqueeze(1)
			ed = self.w.new(length - npos, self.num_dim)
		ed[:, 0::2], ed[:, 1::2] = torch.sin(pos * self.rdiv_term), torch.cos(pos * self.rdiv_term)

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

	def __init__(self, isize, hsize, osize, num_head=8, dropout=0.0, sparsenorm=False):

		super(MultiHeadAttn, self).__init__()

		self.attn_dim = hsize // num_head
		self.hsize = self.attn_dim * num_head
		self.num_head = num_head

		self.query_adaptor = nn.Linear(isize, self.hsize)
		self.value_adaptor = nn.Linear(isize, self.hsize)
		self.key_adaptor = nn.Linear(isize, self.hsize)

		self.outer = nn.Linear(self.hsize, osize)

		#self.normer = MHSparseNormer(num_head, dim=-1) if sparsenorm else nn.Softmax(dim=-1)
		self.normer = SparseNormer(dim=-1) if sparsenorm else nn.Softmax(dim=-1)

		self.drop = nn.Dropout(dropout) if dropout > 0.0 else None

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
		# real_iK: MultiHead iK (bsize, seql, vsize) => (bsize, nheads, seql, adim)
		# real_iV: MultiHead iV (bsize, seql, vsize) => (bsize, nheads, seql, adim)

		real_iQ, real_iK, real_iV = self.query_adaptor(iQ).view(bsize, nquery, nheads, adim).transpose(1, 2), self.key_adaptor(iK).view(bsize, seql, nheads, adim).transpose(1, 2), self.value_adaptor(iV).view(bsize, seql, nheads, adim).transpose(1, 2)

		# scores (bsize, nheads, nquery, adim) * (bsize, nheads, seql, adim)' => (bsize, nheads, nquery, seql)

		scores = torch.div(torch.matmul(real_iQ, real_iK.transpose(2, 3)), sqrt(adim))

		if mask is not None:
			scores.masked_fill_(torch.unsqueeze(mask, 1).expand_as(scores), -1e32)

		scores = self.normer(scores)

		if self.drop is not None:
			scores = self.drop(scores)

		# oMA: output of MultiHeadAttention T((bsize, nheads, nquery, seql) * (bsize, nheads, seql, adim)) => (bsize, nquery, nheads, adim)

		oMA = torch.matmul(scores, real_iV).transpose(1, 2).contiguous()

		# output of this layer (bsize, nquery, nheads, adim) => (bsize, nquery, osize)

		return self.outer(oMA.view(bsize, nquery, self.hsize))

# Average Attention is proposed in Accelerating Neural Transformer via an Average Attention Network(https://arxiv.org/abs/1805.00631)

class AverageAttn(nn.Module):

	# isize: input size of Feed-forward NN
	# hsize: hidden size of Feed-forward NN
	# dropout: dropout rate for Feed-forward NN
	# num_pos: maximum length of sentence cached, extended length will be generated while needed and droped immediately after that

	def __init__(self, isize, hsize=None, dropout=0.0, num_pos=512):

		super(AverageAttn, self).__init__()

		_hsize = isize if hsize is None else hsize

		self.num_pos = num_pos
		self.register_buffer('w', torch.Tensor(num_pos, num_pos))

		self.ffn = PositionwiseFF(isize, _hsize, dropout, False)

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
			avg = torch.matmul(attn.unsqueeze(0).expand(bsize, seql, seql), iV)

		avg = self.ffn(avg)

		ifg = torch.sigmoid(self.gw(torch.cat((iQ, avg), -1)))
		isize = avg.size(-1)
		igate = ifg.narrow(-1, 0, isize)
		fgate = ifg.narrow(-1, isize, isize)

		return igate * iQ + fgate * avg

	def reset_parameters(self):

		self.w = self.get_ext(self.num_pos)

	def get_ext(self, npos):

		_tmp = (1.0 / torch.arange(1, npos + 1, dtype=self.w.dtype, device=self.w.device)).unsqueeze(1).repeat(1, npos)

		return torch.tril(_tmp, 0)

def freeze_module(module):

	for p in module.parameters():
		if p.requires_grad:
			p.requires_grad_(False)

def unfreeze_module(module):

	def unfreeze_fixing(mod):
		if "fix_unfreeze" in dir(mod):
			mod.fix_unfreeze()

	for p in module.parameters():
		p.requires_grad_(True)

	module.apply(unfreeze_fixing)

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

# Actually ATR from: Simplifying Neural Machine Translation with Addition-Subtraction Twin-Gated Recurrent Networks
class GatedCombiner(nn.Module):

	# isize: input size of Feed-forward NN

	def __init__(self, isize):

		super(GatedCombiner, self).__init__()

		self.t1 = nn.Linear(isize, isize)
		self.t2 = nn.Linear(isize, isize)

	# x: input to the cell
	# cell: cell to update

	def forward(self, x, cell):

		p, q = self.t1(x), self.t2(cell)

		igate, fgate = torch.sigmoid(p + q), torch.sigmoid(p - q)

		return igate * p + fgate * q

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
		self.register_buffer("w", ((torch.sigmoid((torch.arange(1, xseql + 1, dtype=torch.float, requires_grad=False) / self.wstep)) * 2 - 1) * self.tarv).unsqueeze(0).unsqueeze(-1))

	def forward(self, x, expand=True):

		seql = x.size(1)

		out = self.get_ext(seql) if seql > self.xseql else self.w.narrow(1, 0, seql)

		return out.expand_as(x) if expand else out

	def get_ext(self, seql):

		_tmp = ((torch.sigmoid((torch.arange(self.xseql + 1, seql + 1, dtype=self.w.dtype, device=self.w.device, requires_grad=False) / self.wstep)) * 2.0 - 1.0) * self.tarv).unsqueeze(0).unsqueeze(-1)

		return torch.cat((self.w, _tmp), 1)

class ApproximateEmb(nn.Module):

	def __init__(self, weight):

		super(ApproximateEmb, self).__init__()
		self.weight = weight

	def forward(self, inpute):

		isize = list(inpute.size())
		out = inpute.view(-1, isize[-1])
		out = torch.mm(out, self.weight)
		isize[-1] = -1
		return out.view(isize)

# 2 kinds of GELU activation function implementation according to https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py#L53-L58

class GeLU_GPT(nn.Module):

	def __init__(self):

		super(GeLU_GPT, self).__init__()

		self.k = sqrt(2.0 / pi)

	def forward(self, x):

		return 0.5 * x * (1.0 + torch.tanh(self.k * (x + 0.044715 * torch.pow(x, 3))))

class GeLU_BERT(nn.Module):

	def __init__(self):

		super(GeLU_BERT, self).__init__()

		self.k = sqrt(2.0)

	def forward(self, x):

		return 0.5 * x * (1.0 + torch.erf(x / self.k))

# SparseNormer is proposed in GLoMo: Unsupervisedly Learned Relational Graphs as Transferable Representations(https://arxiv.org/abs/1806.05662)

class SparseNormer(nn.Module):

	# dim: dimension to normalize

	def __init__(self, dim=-1, ieps=1e-32):

		super(SparseNormer, self).__init__()

		self.dim = dim
		self.bias = nn.Parameter(torch.zeros(1))
		self.act = nn.ReLU()
		self.ieps = ieps

	def forward(self, x, temp=None):

		_tmp = self.act(x + self.bias)
		_tmp = _tmp * _tmp# if temp is None else torch.pow(_tmp, temp)

		# fix zero-devision in case all elements in _tmp are 0.
		return _tmp / (_tmp.sum(self.dim, keepdim=True) + self.ieps)

class MHSparseNormer(nn.Module):

	# nheads: number of heads
	# dim: dimension to normalize

	def __init__(self, nheads, dim=-1, ieps=1e-32):

		super(MHSparseNormer, self).__init__()

		self.dim = dim
		self.bias = nn.Parameter(torch.zeros(1, nheads, 1, 1))
		self.act = nn.ReLU()
		self.ieps = ieps

	# input should be: (bsize, nheads, nquery, seql)
	def forward(self, x, temp=None):

		_tmp = self.act(x + self.bias)
		_tmp = _tmp * _tmp# if temp is None else torch.pow(_tmp, temp)

		# fix zero-devision in case all elements in _tmp are 0.
		return _tmp / (_tmp.sum(self.dim, keepdim=True) + self.ieps)

	def fix_init(self):

		self.bias.data.zero_()

class Scorer(nn.Module):

	def __init__(self, isize):

		super(Scorer, self).__init__()

		self.w = nn.Parameter(torch.randn(isize))
		self.bias = nn.Parameter(torch.zeros(1))

	def forward(self, x):

		xsize = x.size()

		out = torch.addmv(self.bias, x.view(-1, xsize[-1]), self.w)

		rsize = list(xsize)
		rsize[-1] = 1

		return out.view(rsize)

class Temperature(nn.Module):

	def __init__(self, isize, minv = 0.125):

		super(Temperature, self).__init__()

		self.w = nn.Parameter(torch.randn(isize))
		self.bias = nn.Parameter(torch.zeros(1))
		self.act = nn.Tanh()
		self.k = nn.Parameter(torch.ones(1))
		self.minv = minv

	def forward(self, x):

		xsize = x.size()

		out = torch.addmv(self.bias, x.view(-1, xsize[-1]), self.w)

		xsize = list(xsize)
		xsize[-1] = 1

		return ((torch.abs(self.k) + self.minv) * (self.act(out) + 1)).view(xsize)

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
		self.register_buffer('w', torch.Tensor(num_steps, num_pos, num_dim))
		self.register_buffer("rdiv_term", torch.exp(torch.arange(dim_offset, num_dim + dim_offset, 2, dtype=self.w.dtype, device=self.w.device) * -(log(10000.0) / num_dim)))
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
		self.w[:, :, 0::2], self.w[:, :, 1::2] = torch.sin(pos * self.rdiv_term) + torch.sin(step * self.rdiv_term), torch.cos(pos * self.rdiv_term) + torch.cos(step * self.rdiv_term)

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
		ed[:, 0::2], ed[:, 1::2] = torch.sin(_pos * self.rdiv_term) + torch.sin(_step * self.rdiv_term), torch.cos(_pos * self.rdiv_term) + torch.cos(_step * self.rdiv_term)

		return ed

	# step of weight to retrieve, start from 0

	def get_pos(self, step, layer):

		return self.w[layer][step] if step < self.num_pos and layer < self.num_steps else self.get_ext(step, layer, True).squeeze(0)

class Noiser(nn.Module):

        def __init__(self, power):

                super(Noiser, self).__init__()
                self.power = power

        def forward(self, inpute):

                if self.training:
                        return torch.randn(inpute.size(), dtype=inpute.dtype, device=inpute.device) * (self.power * inpute.data.abs().mean()) + inpute

                return inpute
