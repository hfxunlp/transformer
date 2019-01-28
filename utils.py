#encoding: utf-8

import torch

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

