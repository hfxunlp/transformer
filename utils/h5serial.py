#encoding: utf-8

import h5py
import torch
from collections.abc import Iterator

from utils.fmt.base import dict_is_list, list2dict

from cnfg.ihyp import h5_libver, h5modelwargs, hdf5_track_order, list_key_func

try:
	h5py.get_config().track_order = hdf5_track_order
except Exception as e:
	pass

if hasattr(h5py.File, "__enter__") and hasattr(h5py.File, "__exit__"):
	h5File = h5py.File
else:
	class h5File(h5py.File):

		def __enter__(self):

			return self

		def __exit__(self, *inputs, **kwargs):

			self.close()

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
			_ = _v if _v.device.type == "cpu" else _v.cpu()
			gwrt.create_dataset(k, data=_.numpy(), **h5args)

def h5write_list(gwrt, ltw, h5args=h5modelwargs):

	h5write_dict(gwrt, list2dict(ltw, kfunc=list_key_func), h5args=h5args)

def h5save(obj_save, fname, h5args=h5modelwargs):

	with h5File(fname, "w", libver=h5_libver) as h5f:
		_obj_save = tuple(obj_save) if isinstance(obj_save, Iterator) else obj_save
		if isinstance(_obj_save, dict):
			h5write_dict(h5f, _obj_save, h5args=h5args)
		elif isinstance(_obj_save, (list, tuple,)):
			h5write_list(h5f, _obj_save, h5args=h5args)
		else:
			h5write_list(h5f, [_obj_save], h5args=h5args)

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
			rsd[k] = torch.from_numpy(v[()])
		else:
			rsd[k] = h5load_group(v)

	return rsd

def h5load(fname, restore_list=True):

	with h5File(fname, "r") as f:
		rsd = h5load_group(f)
	if restore_list:
		rsd = restore_list_in_dict(rsd)

	return rsd
