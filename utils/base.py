#encoding: utf-8

import torch
from torch.nn import ModuleDict

from threading import Thread

from functools import wraps

from random import sample
from random import seed as rpyseed

from math import ceil

import logging

from utils.h5serial import h5save, h5load

secure_type_map = {torch.float16: torch.float64, torch.float32: torch.float64, torch.uint8: torch.int64, torch.int8: torch.int64, torch.int16: torch.int64, torch.int32: torch.int64}

# handling torch.bool
if torch.__version__ < "1.2.0":
	mask_tensor_type = torch.uint8
	nccl_type_map = None
else:
	mask_tensor_type = torch.bool
	secure_type_map[mask_tensor_type] = torch.int64
	nccl_type_map = {torch.bool:torch.uint8}

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

def clear_pad(batch_in, mask=None, dim=-1):

	_mask = batch_in.eq(0) if mask is None else mask
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

def getlr(optm):

	lr = []
	for i, param_group in enumerate(optm.param_groups):
		lr.append(float(param_group['lr']))

	return lr

def updated_lr(oldlr, newlr):

	rs = False
	for olr, nlr in zip(oldlr, newlr):
		if olr != nlr:
			rs = True
			break

	return rs

def reset_Adam(optm, amsgrad=False):

	for group in optm.param_groups:
		for p in group['params']:
			state = optm.state[p]
			if len(state) != 0:
				state['step'] = 0
				state['exp_avg'].zero_()
				state['exp_avg_sq'].zero_()
				if amsgrad:
					state['max_exp_avg_sq'].zero_()

def reinit_Adam(optm, amsgrad=False):

	for group in optm.param_groups:
		for p in group['params']:
			optm.state[p].clear()

def dynamic_sample(incd, dss_ws, dss_rm):

	rd = {}
	for k, v in incd.items():
		if v in rd:
			rd[v].append(k)
		else:
			rd[v] = [k]
	incs = list(rd.keys())
	incs.sort(reverse=True)
	_full_rl = []
	for v in incs:
		_full_rl.extend(rd[v])

	return _full_rl[:dss_ws] + sample(_full_rl[dss_ws:], dss_rm) if dss_rm > 0 else _full_rl[:dss_ws]

def load_model_cpu(modf, base_model):

	mpg = h5load(modf)

	for para, mp in zip(base_model.parameters(), mpg):
		para.data = mp.data

	return base_model

def load_model_cpu_old(modf, base_model):

	base_model.load_state_dict(h5load(modf))

	return base_model

def save_model(model, fname, sub_module=False, logger=None):

	_msave = model.module if sub_module else model
	try:
		h5save([t.data for t in _msave.parameters()], fname)
	except Exception as e:
		if logger is None:
			print(e)
		else:
			logger.info(str(e))

def async_save_model(model, fname, sub_module=False, logger=None, para_lock=None, log_success=None):

	def _worker(model, fname, sub_module=False, logger=None, para_lock=None, log_success=None):

		success = True
		_msave = model.module if sub_module else model
		try:
			if para_lock is None:
				h5save([t.data for t in _msave.parameters()], fname)
			else:
				with para_lock:
					h5save([t.data for t in _msave.parameters()], fname)
		except Exception as e:
			if logger is None:
				print(e)
			else:
				logger.info(str(e))
			success = False
		if success and (logger is not None) and (log_success is not None):
			logger.info(log_success)

	Thread(target=_worker, args=(model, fname, sub_module, logger, para_lock, log_success)).start()

def get_logger(fname):

	logger = logging.getLogger(__name__)
	logger.setLevel(level = logging.INFO)
	handler = logging.FileHandler(fname)
	handler.setLevel(logging.INFO)
	formatter = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s')
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
		# Make cudnn methods deterministic according to: https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/utils/misc.py#L80-L82
		torch.backends.cudnn.deterministic = True

def repeat_bsize_for_beam_tensor(tin, beam_size):

	_tsize = list(tin.size())
	_rarg = [beam_size if i == 1 else 1 for i in range(len(_tsize))]
	_tsize[0] *= beam_size
	_tout = tin.repeat(*_rarg).view(_tsize)

	return _tout

def expand_bsize_for_beam(*inputs, beam_size=1):

	outputs = []
	for inputu in inputs:
		if inputu is None:
			outputs.append(None)
		elif isinstance(inputu, list):
			outputs.append(list(expand_bsize_for_beam(*inputu, beam_size=beam_size)))
		elif isinstance(inputu, tuple):
			outputs.append(tuple(expand_bsize_for_beam(*inputu, beam_size=beam_size)))
		elif isinstance(inputu, dict):
			_tmp = {}
			for _k, _v in inputu.items():
				_tmp[_k] = expand_bsize_for_beam(_v, beam_size=beam_size)
			outputs.append(_tmp)
		else:
			outputs.append(repeat_bsize_for_beam_tensor(inputu, beam_size))

	return outputs[0] if len(inputs) == 1 else tuple(outputs)

def remove_layers(all_layers, ltr):

	rs = []
	for i, _l in enumerate(all_layers):
		if i not in ltr:
			rs.append(_l)

	return rs

def free_cache(free_cuda=False):

	if free_cuda:
		torch.cuda.empty_cache()

def filter_para_grad(plin):

	rs = []
	for para in plin:
		if para.requires_grad:
			rs.append(para)

	return rs

def ModuleList2Dict(modin):

	return ModuleDict(zip([str(i) for i in range(len(modin))], modin))

def add_module(m, strin, m_add):

	_name_list = strin.split(".")
	if len(_name_list) == 1:
		m.add_module(strin, m_add)
	else:
		_m = m
		# update _modules with pytorch: https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module.add_module
		for _tmp in _name_list[:-1]:
			_m = _m._modules[_tmp]
		_m.add_module(_name_list[-1], m_add)

	return m

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
