#encoding: utf-8

import torch

non_tensor = torch.Tensor()

def transfer_CNone_tuple(lin):

	return tuple(non_tensor if lu is None else lu for lu in lin)

def transfer_CNone_list(lin):

	return [non_tensor if lu is None else lu for lu in lin]

def transfer_CNone(din):

	if isinstance(din, list):
		return [transfer_CNone(du) for du in din]
	elif isinstance(din, tuple):
		return tuple(transfer_CNone(du) for du in din)
	elif isinstance(din, dict):
		return {k: transfer_CNone(du) for k, du in din.items()}
	else:
		return non_tensor if din is None else din
