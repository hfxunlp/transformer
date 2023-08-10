#encoding: utf-8

import torch
from torch import nn
from torch.autograd import Function

from utils.torch.ext import multinomial

class SampleMaxFunction(Function):

	@staticmethod
	def forward(ctx, x, dim=-1, keepdim=False):

		out = multinomial(x, 1, replacement=True, dim=dim)

		return out if keepdim else out.squeeze(dim)

	@staticmethod
	def backward(ctx, grad_outputs):

		return None, None, None

SampleMaxFunc = SampleMaxFunction.apply

class RetrievalFunction(Function):

	# attnmat: (bsize, nquery, seql)
	# vmat: (bsize, seql, isize)
	@staticmethod
	def forward(ctx, attnmat, vmat):

		bsize, nquery, seql = attnmat.size()
		isize = vmat.size(-1)

		# _ind: (bsize, nquery)
		_ind = multinomial(attnmat, 1, replacement=True, dim=-1).squeeze(-1)

		_ind += torch.arange(0, bsize * seql, seql, dtype=_ind.dtype, device=_ind.device).view(bsize, 1)
		_ind = _ind.view(bsize * nquery)

		ctx.save_for_backward(_ind, vmat)#, _mask

		return vmat.view(bsize * seql, isize).index_select(0, _ind).view(bsize, nquery, isize)

	# grad_outputs: (bsize, nquery, isize)
	@staticmethod
	def backward(ctx, grad_outputs):

		if (grad_outputs is not None) and (ctx.needs_input_grad[0] or ctx.needs_input_grad[1]):
			_ind, _vmat = ctx.saved_tensors#, _mask

			_grad_outputs = grad_outputs.contiguous()

			if ctx.needs_input_grad[1]:
				bsize, seql, isize = _vmat.size()
				grad_vmat = _grad_outputs.new_zeros(bsize * seql, isize).index_add_(0, _ind, _grad_outputs.view(-1, isize)).view(bsize, seql, isize)
			else:
				grad_vmat = None

			return _grad_outputs.bmm(_vmat.transpose(1, 2)) if ctx.needs_input_grad[0] else None, grad_vmat#.masked_fill_(~_mask, 0.0)
		else:
			return None, None

RetrievalFunc = RetrievalFunction.apply

class Retriever(nn.Module):

	def forward(self, attnmat, vmat, **kwargs):

		_idim = attnmat.dim()
		if _idim > 3:
			_attnsize = attnmat.size()
			_rsize = list(vmat.size())
			_attnmat, _vmat = attnmat.view(-1, *_attnsize[_idim - 2:]), vmat.view(-1, *_rsize[_idim - 2:])
			_rsize[-2] = _attnsize[-2]
		else:
			_attnmat, _vmat, _rsize = attnmat, vmat, None
		if self.training:
			rs = RetrievalFunc(_attnmat, _vmat)
		else:
			bsize, nquery, seql = _attnmat.size()
			isize = _vmat.size(-1)
			# _ind: (bsize, nquery)
			_ind = _attnmat.argmax(-1)
			_ind += torch.arange(0, bsize * seql, seql, dtype=_ind.dtype, device=_ind.device).view(bsize, 1)
			rs = _vmat.view(bsize * seql, isize).index_select(0, _ind.view(bsize * nquery)).view(bsize, nquery, isize)

		return rs if _rsize is None else rs.view(_rsize)

class SamplerFunction(Function):

	@staticmethod
	def forward(ctx, inputs, dim=-1, bsize=None):

		_ics = inputs.cumsum(dim)
		isize = list(inputs.size())
		isize[dim] = 1
		if bsize is None:
			_sv = inputs.new_empty(isize).uniform_(0.0, 1.0)
			ctx.sum_batch = False
		else:
			isize.insert(0, bsize)
			_sv = inputs.new_empty(isize).uniform_(0.0, 1.0)
			_ics = _ics.unsqueeze(0)
			ctx.sum_batch = True
		_ms = _ics.ge(_sv).int().cumsum(dim).eq(1)

		return _ms.to(inputs.dtype, non_blocking=True)

	@staticmethod
	def backward(ctx, grad_outputs):

		return (grad_outputs.sum(0) if ctx.sum_batch else grad_outputs) if (grad_outputs is not None) and ctx.needs_input_grad[0] else None, None, None

SamplerFunc = SamplerFunction.apply

class EffSamplerFunction(Function):

	# inputs: bsize, nop, ...
	# weight: (bsize,) nop
	@staticmethod
	def forward(ctx, inputs, weight, dim=-1, add_bdim=False):

		isize = inputs.size()
		bsize, nop = isize[:2]
		isize = list(isize)

		_ics = weight.cumsum(dim)
		wsize = list(weight.size())
		wsize[dim] = 1
		if add_bdim:
			_ics = _ics.unsqueeze(0)
			wsize.insert(0, bsize)
			ctx.sum_batch = True
		else:
			ctx.sum_batch = False
		_sv = weight.new_empty(wsize).uniform_(0.0, 1.0)
		_mask = _ics.ge(_sv).int().cumsum(dim).eq(1)
		_ind = _mask.argmax(dim)

		_ind += torch.arange(0, bsize * nop, nop, dtype=_ind.dtype, device=_ind.device)

		ctx.save_for_backward(_ind, inputs)#, _mask

		del isize[1]

		return inputs.view(bsize * nop, -1).index_select(0, _ind).view(isize)

	# grad_outputs: bsize, ...
	@staticmethod
	def backward(ctx, grad_outputs):

		if (grad_outputs is not None) and (ctx.needs_input_grad[0] or ctx.needs_input_grad[1]):
			_ind, inputs = ctx.saved_tensors#, _mask
			isize = inputs.size()
			bsize, nop = isize[:2]

			if ctx.needs_input_grad[1]:
				grad_weight = (grad_outputs.unsqueeze(1) * inputs).view(bsize, nop, -1).sum(-1)#.masked_fill_(~_mask, 0.0)
				if ctx.sum_batch:
					grad_weight = grad_weight.sum(0)
			else:
				grad_weight = None

			return grad_outputs.new_zeros(isize).view(bsize * nop, -1).index_copy_(0, _ind, grad_outputs.view(bsize, -1)).view(isize) if ctx.needs_input_grad[0] else None, grad_weight, None, None
		else:
			return None, None, None, None

EffSamplerFunc = EffSamplerFunction.apply

class Sampler(nn.Module):

	def __init__(self, dim=-1, **kwargs):

		super(Sampler, self).__init__()

		self.dim = dim

	def forward(self, inputs, bsize=None, **kwargs):

		return SamplerFunc(inputs, self.dim, bsize)

class EffSampler(Sampler):

	def forward(self, inputs, weight, add_bdim=False, **kwargs):

		return EffSamplerFunc(inputs, weight, self.dim, add_bdim)
