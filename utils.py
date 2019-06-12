#encoding: utf-8

import torch
from torch.nn.init import xavier_uniform_, kaiming_uniform_
from torch.nn import Embedding, Linear, LayerNorm

from math import sqrt

from random import sample
from random import seed as rpyseed

import logging

def pad_tensors(tensor_list):

	def get_pad_size(tsize, stdlen):

		nsize = list(tsize)
		nsize[-1] = stdlen - tsize[-1]

		return nsize

	maxlen = 0
	for tensor in tensor_list:
		tlen = tensor.size(-1)
		if tlen > maxlen:
			maxlen = tlen

	return [tensor if tensor.size(-1) == maxlen else torch.cat((tensor, tensor.new_zeros(get_pad_size(tensor.size(), maxlen))), -1) for tensor in tensor_list]

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

	mpg = torch.load(modf, map_location='cpu')

	for para, mp in zip(base_model.parameters(), mpg):
		para.data = mp.data

	return base_model

def load_model_cpu_old(modf, base_model):

	base_model.load_state_dict(torch.load(modf, map_location='cpu'))

	return base_model

def save_model(model, fname, sub_module):

	if sub_module:
		torch.save([t.data for t in model.module.parameters()], fname)
	else:
		torch.save([t.data for t in model.parameters()], fname)

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

def init_model_params_glorot(modin, hyp=None):

	_scale = sqrt(1.0 / 3.0) if hyp is None else hyp

	for p in modin.parameters():
		if p.requires_grad and (p.dim() > 1):
			xavier_uniform_(p, gain=_scale)

	return modin

def init_model_params_kaiming(modin, hyp=None):

	_scale = sqrt(5.0) if hyp is None else hyp

	for p in modin.parameters():
		if p.requires_grad and (p.dim() > 1):
			kaiming_uniform_(p, a=_scale)

	return modin

def init_model_params(modin, scale_glorot=None, scale_kaiming=None):

	_tmpm = init_model_params_kaiming(modin, scale_kaiming)

	for _m in _tmpm.modules():
		if isinstance(_m, Embedding):
			init_model_params_glorot(_m, scale_glorot)
		elif isinstance(_m, Linear):
			if _m.bias is not None:
				with torch.no_grad():
					_m.bias.zero_()
		elif isinstance(_m, LayerNorm):
			with torch.no_grad():
				_m.weight.fill_(1.0)
				_m.bias.zero_()

	return _tmpm

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
