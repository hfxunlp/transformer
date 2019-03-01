#encoding: utf-8

import torch
from torch.nn.init import xavier_uniform_

from random import sample

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

def init_model_params(modin):

	for p in modin.parameters():
		if p.requires_grad and (p.dim() > 1):
			xavier_uniform_(p)
	return modin
