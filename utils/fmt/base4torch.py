#encoding: utf-8

import torch
from math import sqrt

from utils.fmt.base import list_reader
from utils.h5serial import h5load
from utils.torch.comp import torch_no_grad

def parse_cuda(use_cuda_arg, gpuid=None):

	if use_cuda_arg and torch.cuda.is_available():
		use_cuda = True
		ngpus = torch.cuda.device_count()
		if gpuid is None:
			cuda_devices = tuple(torch.device("cuda", i) for i in range(ngpus))
		else:
			gpuid = tuple(int(_.strip()) for _ in gpuid[gpuid.find(":") + 1:].split(","))
			cuda_devices = tuple(torch.device("cuda", i) for i in gpuid if (i >= 0) and (i < ngpus))
		cuda_device = cuda_devices[0]
		if len(cuda_devices) > 1:
			multi_gpu = True
		else:
			cuda_devices = None
			multi_gpu = False
		torch.cuda.set_device(cuda_device.index)
	else:
		use_cuda, cuda_device, cuda_devices, multi_gpu = False, None, None, False

	return use_cuda, cuda_device, cuda_devices, multi_gpu

def parse_cuda_decode(use_cuda_arg, gpuid=None, multi_gpu_decoding=False):

	if use_cuda_arg and torch.cuda.is_available():
		use_cuda = True
		ngpus = torch.cuda.device_count()
		if gpuid is None:
			cuda_devices = tuple(torch.device("cuda", i) for i in range(ngpus))
		else:
			gpuid = tuple(int(_.strip()) for _ in gpuid[gpuid.find(":") + 1:].split(","))
			cuda_devices = tuple(torch.device("cuda", i) for i in gpuid if (i >= 0) and (i < ngpus))
		if len(cuda_devices) > 1 and multi_gpu_decoding:
			cuda_device = cuda_devices[0]
			multi_gpu = True
		else:
			cuda_device = cuda_devices[-1]
			cuda_devices = None
			multi_gpu = False
		torch.cuda.set_device(cuda_device.index)
	else:
		use_cuda, cuda_device, cuda_devices, multi_gpu = False, None, None, False

	return use_cuda, cuda_device, cuda_devices, multi_gpu

def load_emb_txt(vcb, embf):

	rs = {}
	for tmp in list_reader(embf, keep_empty_line=False):
		wd = tmp[0]
		if wd in vcb or wd == "<unk>":
			rs[wd] = torch.as_tensor([float(_t) for _t in tmp[1:]])

	return rs

def load_emb(embf, embt, nword, scale_down_emb, freeze_emb):

	_emb = h5load(embf)
	if nword < _emb.size(0):
		_emb = _emb.narrow(0, 0, nword).contiguous()
	if scale_down_emb:
		_emb.div_(sqrt(embt.size(-1)))
	with torch_no_grad():
		embt.copy_(_emb)
	if freeze_emb:
		embt.requires_grad_(False)
	else:
		embt.requires_grad_(True)

	return embt
