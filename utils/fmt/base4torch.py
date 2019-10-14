#encoding: utf-8

import torch

from math import sqrt
from utils.fmt.base import list_reader

def parse_cuda(use_cuda_arg, gpuid):

	if use_cuda_arg and torch.cuda.is_available():
		use_cuda = True
		if len(gpuid.split(",")) > 1:
			cuda_device = torch.device(gpuid[:gpuid.find(",")].strip())
			cuda_devices = [int(_.strip()) for _ in gpuid[gpuid.find(":") + 1:].split(",")]
			multi_gpu = True
		else:
			cuda_device = torch.device(gpuid)
			cuda_devices = None
			multi_gpu = False
		torch.cuda.set_device(cuda_device.index)
		#torch.backends.cudnn.benchmark = True
	else:
		use_cuda = False
		cuda_device = False
		cuda_devices = None
		multi_gpu = False

	return use_cuda, cuda_device, cuda_devices, multi_gpu

def parse_cuda_decode(use_cuda_arg, gpuid, multi_gpu_decoding):

	if use_cuda_arg and torch.cuda.is_available():
		use_cuda = True
		if len(gpuid.split(",")) > 1:
			if multi_gpu_decoding:
				cuda_device = torch.device(gpuid[:gpuid.find(",")].strip())
				cuda_devices = [int(_.strip()) for _ in gpuid[gpuid.find(":") + 1:].split(",")]
				multi_gpu = True
			else:
				cuda_device = torch.device("cuda:" + gpuid[gpuid.rfind(",") + 1:].strip())
				cuda_devices = None
				multi_gpu = False
		else:
			cuda_device = torch.device(gpuid)
			cuda_devices = None
			multi_gpu = False
		torch.cuda.set_device(cuda_device.index)
		#torch.backends.cudnn.benchmark = True
	else:
		use_cuda = False
		cuda_device = False
		cuda_devices = None
		multi_gpu = False

	return use_cuda, cuda_device, cuda_devices, multi_gpu

def load_emb_txt(vcb, embf):

	rs = {}
	for tmp in list_reader(embf):
		wd = tmp[0]
		if wd in vcb or wd == "<unk>":
			rs[wd] = torch.tensor([float(_t) for _t in tmp[1:]])

	return rs

def load_emb(embf, embt, nword, scale_down_emb, freeze_emb):

	_emb = torch.load(embf, map_location='cpu')
	if nword < _emb.size(0):
		_emb = _emb.narrow(0, 0, nword).contiguous()
	if scale_down_emb:
		_emb.div_(sqrt(embt.size(-1)))
	with torch.no_grad():
		embt.copy_(_emb)
	if freeze_emb:
		embt.requires_grad_(False)
	else:
		embt.requires_grad_(True)

	return embt
