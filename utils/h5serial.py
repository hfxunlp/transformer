#encoding: utf-8

import torch, h5py

from collections.abc import Iterator

from utils.fmt.base import list2dict, dict_is_list

from cnfg.ihyp import *

def h5write_dict(gwrt, dtw, h5args=h5modelwargs):

	for k, v in dtw.items():
		_v = tuple(v) if isinstance(v, Iterator) else v
		if isinstance(_v, dict):
			gwrt.create_group(k)
			h5write_dict(gwrt[k], _v, h5args=h5args)
		elif isinstance(_v, (list, tuple,)):
			gwrt.create_group(k)
			h5write_list(gwrt[k], _v, h5args=h5args)
		else:
			if _v.device.type == 'cpu':
				gwrt.create_dataset(k, data=_v.numpy(), **h5args)
			else:
				gwrt.create_dataset(k, data=_v.cpu().numpy(), **h5args)

def h5write_list(gwrt, ltw, h5args=h5modelwargs):

	h5write_dict(gwrt, list2dict(ltw, kfunc=list_key_func), h5args=h5args)

def h5save(obj_save, fname, h5args=h5modelwargs):

	h5f = h5py.File(fname, 'w')
	_obj_save = tuple(obj_save) if isinstance(obj_save, Iterator) else obj_save
	if isinstance(_obj_save, dict):
		h5write_dict(h5f, _obj_save, h5args=h5args)
	elif isinstance(_obj_save, (list, tuple,)):
		h5write_list(h5f, _obj_save, h5args=h5args)
	else:
		h5write_list(h5f, [_obj_save], h5args=h5args)
	h5f.close()

def restore_list_in_dict(din):

	if isinstance(din, dict):
		_key_set = set(din.keys())
		if dict_is_list(_key_set, kfunc=list_key_func):
			return [restore_list_in_dict(din[list_key_func(i)]) for i in range(len(_key_set))]
		else:
			return {k: restore_list_in_dict(v) for k, v in din.items()}
	else:
		return din

def h5load_group(grd):

	rsd = {}
	for k, v in grd.items():
		if isinstance(v, h5py.Dataset):
			rsd[k] = torch.from_numpy(v[:])
		else:
			rsd[k] = h5load_group(v)
	return rsd

def h5load(fname, restore_list=True):

	f = h5py.File(fname, "r")
	rsd = h5load_group(f)
	f.close()
	if restore_list:
		rsd = restore_list_in_dict(rsd)
	return rsd
