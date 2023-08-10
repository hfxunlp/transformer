#encoding: utf-8

import torch

from utils.fmt.parser import parse_none

non_tensor = torch.Tensor()

def transfer_CNone_tuple(lin):

	return tuple(parse_none(_, non_tensor) for _ in lin)

def transfer_CNone_list(lin):

	return [parse_none(_, non_tensor) for _ in lin]

def transfer_CNone(din):

	if isinstance(din, list):
		return [transfer_CNone(du) for du in din]
	elif isinstance(din, tuple):
		return tuple(transfer_CNone(du) for du in din)
	elif isinstance(din, dict):
		return {k: transfer_CNone(du) for k, du in din.items()}
	else:
		return parse_none(din, non_tensor)
