#encoding: utf-8

import torch
from math import sqrt
from torch import nn

from utils.torch.comp import torch_no_grad

from cnfg.ihyp import bind_c_forward, use_c_backend_group

class GroupLinear(nn.Module):

	# isize: input dimension (dimension per group * ngroup)
	# osize: output dimension (dimension per group * ngroup)
	# ngroup: number of group
	# bias: enable bias or not
	# trans_input: split input into groups before computing
	# shuffle: shuffle across groups for output
	# flatten_output: concatenate outputs of groups

	def __init__(self, isize, osize, ngroup, bias=True, trans_input=True, shuffle=False, flatten_output=True, **kwargs):

		super(GroupLinear, self).__init__()

		self.ngroup, self.shuffle, self.flatten_output, self.trans_input = ngroup, shuffle, flatten_output, trans_input
		self.del_gdim = self.flatten_output and (not self.trans_input)
		self.i_gdim = self.trans_input and (not self.flatten_output)
		self.isize = isize // ngroup
		self.osize = osize // ngroup

		self.weight = nn.Parameter(torch.Tensor(ngroup, self.isize, self.osize).uniform_(- sqrt(1.0 / self.isize), sqrt(1.0 / self.isize)))
		if bias:
			self.bias = nn.Parameter(torch.zeros(ngroup, 1, self.osize))
		else:
			self.bias = None

		if self.c_available():
			self.c_init()

	# inputu: (..., isize)

	def forward(self, inputu, weight=None, bias=None, **kwargs):

		_size = list(inputu.size())
		_id = inputu.view(-1, self.ngroup, self.isize if self.trans_input else _size[-1]) if (inputu.dim() != 3) or self.trans_input else inputu
		_id = _id.transpose(0, 1)

		_weight = self.weight if weight is None else weight
		_bias = self.bias if bias is None else bias

		# out: (ngroup, bsize, isize) * (ngroup, isize, osize) => (ngroup, bsize, osize)
		out = _id.bmm(_weight) if _bias is None else _bias.baddbmm(_id, _weight)
		# out: (bsize, osize, ngroup) if self.shuffle else (bsize, ngroup, osize)
		out = out.permute(1, 2, 0) if self.shuffle else out.transpose(0, 1)

		_size[-1] = -1
		if self.i_gdim:
			_size.insert(-1, self.ngroup)
		elif self.del_gdim:
			del _size[-2]

		return out.contiguous().view(_size)

	def extra_repr(self):
		return "groups={}, in_features={}, out_features={}, bias={}".format(self.ngroup, self.ngroup * self.isize, self.ngroup * self.osize, self.bias is not None)

	def fix_init(self):

		with torch_no_grad():
			self.weight.uniform_(- sqrt(1.0 / self.isize), sqrt(1.0 / self.isize))
			if self.bias is not None:
				self.bias.zero_()

	def c_available(self):

		return use_c_backend_group and (type(self) == GroupLinear)

	def c_init(self, bind=bind_c_forward):

		try:
			try:
				import group_cpp
			except Exception as e:
				from torch.utils.cpp_extension import load
				group_cpp = load(name="group_cpp", sources=["modules/cpp/group/group.cpp", "modules/cpp/group/group_func.cpp"])
			self.c_forward_func = group_cpp.forward
		except:
			self.c_forward_func = None
		if self.c_forward_func is not None:
			self.c_build_cache()
			if bind:
				GroupLinear.forward = GroupLinear.c_forward

	def c_forward(self, inputu, weight=None, bias=None):

		return self.c_forward_func(*self.c_build_inputs(inputu, weight=weight, bias=bias))

	def c_build_cache(self):

		self.aargs = ({"isize": self.isize, "ngroup": self.ngroup}, {"trans_input": self.trans_input, "shuffle": self.shuffle, "i_gdim": self.i_gdim, "del_gdim": self.del_gdim},)
		self.targs = dict(self.named_parameters())

	def c_build_inputs(self, inputu, weight=None, bias=None):

		i_d = self.targs.copy()
		i_d["inputu"] = inputu
		if weight is not None:
			i_d["weight"] = weight
		if bias is not None:
			i_d["bias"] = bias

		return i_d, *self.aargs
