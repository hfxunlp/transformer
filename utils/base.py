#encoding: utf-8

import logging
import torch
from functools import wraps
from math import ceil
from os import makedirs
from os.path import exists as fs_check
from random import seed as rpyseed
from torch import Tensor
from torch.nn import ModuleDict

from cnfg.vocab.base import pad_id

def pad_tensors(tensor_list, dim=-1):

	def get_pad_size(tsize, stdlen, dim=-1):

		nsize = list(tsize)
		nsize[dim] = stdlen - tsize[dim]

		return nsize

	maxlen = 0
	for tensor in tensor_list:
		tlen = tensor.size(dim)
		if tlen > maxlen:
			maxlen = tlen

	return [tensor if tensor.size(dim) == maxlen else torch.cat((tensor, tensor.new_zeros(get_pad_size(tensor.size(), maxlen))), dim) for tensor in tensor_list]

def clear_pad(batch_in, mask=None, dim=-1, pad_id=pad_id):

	_mask = batch_in.eq(pad_id) if mask is None else mask
	npad = _mask.int().sum(dim).min().item()
	if npad > 0:
		return batch_in.narrow(dim, 0, batch_in.size(dim) - npad)
	else:
		return batch_in

def clear_pad_mask(batch_list, mask, dims, mask_dim=-1, return_contiguous=True):

	npad = mask.int().sum(mask_dim).min().item()
	if npad > 0:
		_n_ret = mask.size(mask_dim) - npad
		if return_contiguous:
			return [batchu.narrow(dim, 0, _n_ret).contiguous() for batchu, dim in zip(batch_list, dims)], mask.narrow(mask_dim, 0, _n_ret).contiguous()
		else:
			return [batchu.narrow(dim, 0, _n_ret) for batchu, dim in zip(batch_list, dims)], mask.narrow(mask_dim, 0, _n_ret)
	else:
		return batch_list, mask

def eq_indexes(tensor, indexes):

	rs = None
	for ind in indexes:
		if rs is None:
			rs = tensor.eq(ind)
		else:
			rs |= tensor.eq(ind)
	return rs

def get_logger(fname):

	logger = logging.getLogger(__name__)
	logger.setLevel(level=logging.INFO)
	handler = logging.FileHandler(fname)
	handler.setLevel(logging.INFO)
	formatter = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
	handler.setFormatter(formatter)

	console = logging.StreamHandler()
	console.setLevel(logging.INFO)

	logger.addHandler(handler)
	logger.addHandler(console)

	return logger

def set_random_seed(seed, set_cuda=False):

	_rseed = torch.initial_seed() if seed is None else seed
	rpyseed(_rseed)
	torch.manual_seed(_rseed)
	if set_cuda:
		torch.cuda.manual_seed_all(_rseed)

def index_tensors(*inputs, indices=None, dim=0):

	outputs = []
	for inputu in inputs:
		if isinstance(inputu, Tensor):
			outputs.append(inputu.index_select(dim, indices))
		elif isinstance(inputu, dict):
			outputs.append({k: index_tensors(v, indices=indices, dim=dim) for k, v in inputu.items()})
		elif isinstance(inputu, tuple):
			outputs.append(tuple(index_tensors(tmpu, indices=indices, dim=dim) for tmpu in inputu))
		elif isinstance(inputu, list):
			outputs.append([index_tensors(tmpu, indices=indices, dim=dim) for tmpu in inputu])
		else:
			outputs.append(inputu)

	return outputs[0] if len(inputs) == 1 else tuple(outputs)

def select_zero_(x, dim, index):

	x.select(dim, index).zero_()

	return x

def remove_layers(all_layers, ltr):

	return [_l for i, _l in enumerate(all_layers) if i not in ltr]

def free_cache(free_cuda=False):

	if free_cuda:
		torch.cuda.empty_cache()

def filter_para_grad(plin):

	return [para for para in plin if para.requires_grad]

def filter_para_grad_iter(plin):

	for para in plin:
		if para.requires_grad:
			yield para

def ModuleList2Dict(modin):

	return ModuleDict(zip([str(i) for i in range(len(modin))], modin))

def get_module_nl(m, nl):

	_m, _success = m, True
	for _tmp in nl:
		# update _modules with pytorch: https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module.add_module
		if _tmp in _m._modules:
			_m = _m._modules[_tmp]
		else:
			_success = False
			break

	return _m, _success

def add_module(m, strin, m_add, print_func=print, **kwargs):

	_name_list = strin.split(".")
	if len(_name_list) == 1:
		m.add_module(strin, m_add)
	else:
		_m, _success = get_module_nl(m, _name_list[:-1])
		if _success:
			_m.add_module(_name_list[-1], m_add)
		elif print_func is not None:
			print_func(strin)

	return m

def add_parameter(m, strin, p_add, print_func=print, **kwargs):

	_name_list = strin.split(".")
	if len(_name_list) == 1:
		m.register_parameter(strin, p_add)
	else:
		_m, _success = get_module_nl(m, _name_list[:-1])
		if _success:
			_m.register_parameter(_name_list[-1], p_add)
		elif print_func is not None:
			print_func(strin)

	return m

def add_buffer(m, strin, b_add, persistent=True, print_func=print, **kwargs):

	_name_list = strin.split(".")
	if len(_name_list) == 1:
		m.register_buffer(strin, b_add, persistent=persistent)
	else:
		_m, _success = get_module_nl(m, _name_list[:-1])
		if _success:
			_m.register_buffer(_name_list[-1], b_add, persistent=persistent)
		elif print_func is not None:
			print_func(strin)

	return m

def is_buffer_persistent(m, strin, persistent=True, print_func=print, **kwargs):

	_name_list = strin.split(".")
	rs = persistent
	if len(_name_list) == 1:
		# update _non_persistent_buffers_set with pytorch: https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module.register_buffer
		if hasattr(m, "_non_persistent_buffers_set"):
			rs = strin not in m._non_persistent_buffers_set
	else:
		_m, _success = get_module_nl(m, _name_list[:-1])
		if _success:
			if hasattr(_m, "_non_persistent_buffers_set"):
				rs = _name_list[-1] not in _m._non_persistent_buffers_set
		elif print_func is not None:
			print_func(strin)

	return rs

def bind_module_parameter(srcm, tgtm, **kwargs):

	_ = tgtm
	for _n, _p in srcm.named_parameters():
		_ = add_parameter(_, _n, _p, **kwargs)

	return _

def bind_module_buffer(srcm, tgtm, persistent=None, **kwargs):

	_ = tgtm
	for _n, _b in srcm.named_buffers():
		_ = add_buffer(_, _n, _b, persistent=is_buffer_persistent(srcm, _n) if persistent is None else persistent, **kwargs)

	return _

def bind_module_parabuf(srcm, tgtm, persistent=None, **kwargs):

	return bind_module_buffer(srcm, bind_module_parameter(srcm, tgtm, **kwargs), persistent=persistent, **kwargs)

def reduce_model_core(modin, redm, attr_func=None):

	if attr_func is None:
		_m_sel = None
		for _name, _module in modin.named_modules():
			if isinstance(_module, redm):
				if _m_sel is None:
					_m_sel = _module
				else:
					add_module(modin, _name, _m_sel)
	else:
		_m_sel = {}
		for _name, _module in modin.named_modules():
			if isinstance(_module, redm):
				_key = attr_func(_module)
				if _key in _m_sel:
					add_module(modin, _name, _m_sel[_key])
				else:
					_m_sel[_key] = _module

	return modin

def reduce_model_list(modin, redml, attr_funcl=None):

	rsm = modin
	if attr_funcl is None:
		for redm in redml:
			rsm = reduce_model_core(rsm, redm)
	else:
		for redm, attr_func in zip(redml, attr_funcl):
			rsm = reduce_model_core(rsm, redm, attr_func)

	return rsm

def align_modules_by_type(srcml, typ, tgtm):

	srcmi = iter(srcml)
	for _name, _module in tgtm.named_modules():
		if isinstance(_module, typ):
			try:
				_obtm = next(srcmi)
			except Exception:
				_obtm = None
			if _obtm is None:
				break
			else:
				add_module(tgtm, _name, _obtm)

	return tgtm

def report_parameters(modin):

	rs = 0
	for _para in modin.parameters():
		rs += _para.numel()

	return rs

def float2odd(fin):

	_rs = ceil(fin)
	if _rs % 2 == 1:
		_rs -= 1

	return _rs

def wrap_float2odd(func):
	@wraps(func)
	def wrapper(*args, **kwargs):
		return float2odd(func(*args, **kwargs))
	return wrapper

def iternext(iterin):

	try:
		rs = next(iterin)
	except:
		rs = None

	return rs

def divide_para_ind(para_list, ngroup, return_np=False):

	elel = [pu.numel() for pu in para_list]
	n_elel = len(elel)
	if n_elel <= ngroup:
		rs = [(lind, lind + 1,) for lind in range(n_elel)]
	else:
		sum_p = sum(elel)
		p_group = ceil(sum_p / ngroup)
		dp_group = p_group + p_group
		rs = []
		lind = rind = p_g = nd = 0
		nprevs = ngroup - 1
		for elu in elel:
			_np_g = p_g + elu
			if _np_g < p_group:
				rind += 1
				p_g = _np_g
			else:
				if (_np_g + p_g) > dp_group:
					rs.append((lind, rind,))
					lind = rind
					rind += 1
					p_g = elu
				else:
					rind += 1
					rs.append((lind, rind,))
					lind = rind
					p_g = 0
				nd += 1
				if nd >= nprevs:
					break
		rs.append((lind, n_elel,))

	if return_np:
		return rs, [sum(elel[lind:rind]) for lind, rind in rs]
	else:
		return rs

def reorder_by_sort(lins, *inputs, reverse=False):

	td = {}
	for lus, lud in zip(lins, zip(*inputs)):
		if lus in td:
			td[lus].append(lud)
		else:
			td[lus] = [lud]

	rs = None
	for slus in sorted(list(td.keys()), reverse=reverse):
		if rs is None:
			rs = td[slus]
		else:
			rs.extend(td[slus])

	rs = tuple(zip(*rs))

	return rs if len(inputs) > 1 else rs[0]

def range_parameter_iter(model, lind, rind, func=None):

	mp_iter = model.parameters() if func is None else func(model.parameters())
	for i, p in enumerate(mp_iter):
		if i >= lind:
			if i < rind:
				yield p
			else:
				break

def range_parameter_iter_func(model, lind, rind, func=None):

	def iter_func(*args, **kwargs):

		mp_iter = model.parameters() if func is None else func(model.parameters())
		for i, p in enumerate(mp_iter):
			if i >= lind:
				if i < rind:
					yield p
				else:
					break

	return iter_func

def mkdir(pth):

	if not fs_check(pth):
		try:
			makedirs(pth)
		except Exception as e:
			print(e)

class holder(dict):

	def __enter__(self):

		return self

	def get_hold(self, k, sv=None):

		if k in self:
			return self[k]
		else:
			self[k] = sv
			return sv

	def __exit__(self, *inputs, **kwargs):

		pass
