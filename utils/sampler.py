#encoding: utf-8

import torch

def SampleMax(input, dim=-1, keepdim=False):

	_ics = input.cumsum(dim)
	isize = list(input.size())
	isize[dim] = 1
	_sv = input.new_empty(isize).uniform_(0.0, 1.0)
	_ms = _ics.ge(_sv)#.int().cumsum(dim).eq(1)
	_msize = list(_ms.size())
	_nkeep = _msize[dim] - 1
	_msize[dim] = 1
	_ms.logical_xor_(torch.cat((_ms.new_zeros(_msize, dtype=_ms.dtype, device=_ms.device), _ms.narrow(dim, 0, _nkeep),), dim=dim))

	return _ms.byte().argmax(dim=dim, keepdim=keepdim)
