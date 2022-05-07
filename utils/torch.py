#encoding: utf-8

import torch
from numbers import Number

from cnfg.ihyp import ieps_upper_bound_default

upper_one = 1.0 - ieps_upper_bound_default

def bmv(inputm, inputv):

	return inputm.bmm(inputv.unsqueeze(-1)).squeeze(-1)

def randint_t_core(high):

	return high.new_empty(high.size()).uniform_(0.0, upper_one).mul_(high).floor_().to(torch.long)

def randint_t(low, high):

	if isinstance(low, Number) and (low == 0.0):
		return randint_t_core(high)
	else:
		rs = randint_t_core(high - low)
		return rs.add_(low.to(rs.dtype))

def multinomial(x, num_samples, replacement=False, generator=None, dim=-1, **kwargs):

	_ndim = x.dim()

	if (dim == -1) or (dim == (_ndim - 1)):
		_t_output, out = False, x
	else:
		_t_output, out = True, x.transpose(dim, -1)

	if _ndim > 2:
		_osize = list(out.size())
		out = out.view(-1, _osize[-1])
		_osize[-1] = num_samples

	out = out.multinomial(num_samples, replacement=replacement, generator=generator, **kwargs)

	if _ndim > 2:
		out = out.view(_osize)

	if _t_output:
		out = out.transpose(dim, -1)

	return out

def arcsigmoid(x):

	return ((1.0 / x) - 1.0).log().neg()

def arcsoftmax(p):

	_ = p.amax(-1, keepdim=True)

	return (p / _).log()

def ensure_num_threads(n):

	if torch.get_num_threads() < n:
		torch.set_num_threads(n)

	return torch.get_num_threads()

def ensure_num_interop_threads(n):

	if torch.get_num_interop_threads() < n:
		torch.set_num_interop_threads(n)

	return torch.get_num_interop_threads()

class num_threads:

	def __init__(self, n):

		self.num_threads_exe = n

	def __enter__(self):

		self.num_threads_env = torch.get_num_threads()
		torch.set_num_threads(self.num_threads_exe)

	def __exit__(self, *inputs, **kwargs):

		torch.set_num_threads(self.num_threads_env)

class num_interop_threads:

	def __init__(self, n):

		self.num_threads_exe = n

	def __enter__(self):

		self.num_threads_env = torch.get_num_interop_threads()
		torch.set_num_interop_threads(self.num_threads_exe)

	def __exit__(self, *inputs, **kwargs):

		torch.set_num_interop_threads(self.num_threads_env)
