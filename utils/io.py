#encoding: utf-8

import torch
from os import remove
from os.path import exists as fs_check
from threading import Thread

from utils.h5serial import h5load, h5save
from utils.torch.comp import torch_no_grad

from cnfg.ihyp import h5modelwargs, hdf5_save_parameter_name, n_keep_best#, hdf5_load_parameter_name

def load_model_cpu_p(modf, base_model, mp=None, **kwargs):

	with torch_no_grad():
		for para, mp in zip(base_model.parameters(), h5load(modf, restore_list=True) if mp is None else mp):
			para.copy_(mp)

	return base_model

def load_model_cpu_np(modf, base_model, mp=None, strict=False, print_func=print, **kwargs):

	_ = base_model.load_state_dict(h5load(modf, restore_list=False) if mp is None else mp, strict=strict, **kwargs)
	if (print_func is not None) and (_ is not None):
		for _msg in _:
			if _msg:
				print_func(_msg)

	return base_model

def load_model_cpu_auto(modf, base_model, mp=None, **kwargs):

	_mp = h5load(modf, restore_list=True) if mp is None else mp
	_load_model_func = load_model_cpu_p if isinstance(_mp, list) else load_model_cpu_np

	return _load_model_func(modf, base_model, mp=_mp, **kwargs)

mp_func_p = lambda m: [_t.data for _t in m.parameters()]
mp_func_np = lambda m: {_k: _t.data for _k, _t in m.named_parameters()}

load_model_cpu = load_model_cpu_auto#load_model_cpu_np if hdf5_load_parameter_name else load_model_cpu_p
mp_func = mp_func_np if hdf5_save_parameter_name else mp_func_p

class bestfkeeper:

	def __init__(self, fnames=None, k=n_keep_best, **kwargs):

		self.fnames, self.k = [] if fnames is None else fnames, k
		self.clean()

	def update(self, fname=None):

		self.fnames.append(fname)
		self.clean(last_fname=fname)

	def clean(self, last_fname=None):

		_n_files = len(self.fnames)
		_last_fname = (self.fnames[-1] if self.fnames else None) if last_fname is None else last_fname
		while _n_files > self.k:
			fname = self.fnames.pop(0)
			if (fname is not None) and (fname != _last_fname) and fs_check(fname):
				try:
					remove(fname)
				except Exception as e:
					print(e)
			_n_files -= 1

class SaveModelCleaner:

	def __init__(self):

		self.holder = {}

	def __call__(self, fname, typename, **kwargs):

		if typename in self.holder:
			self.holder[typename].update(fname)
		else:
			self.holder[typename] = bestfkeeper(fnames=[fname])

save_model_cleaner = SaveModelCleaner()

def save_model(model, fname, sub_module=False, print_func=print, mtyp=None, h5args=h5modelwargs):

	_msave = model.module if sub_module else model
	try:
		h5save(mp_func(_msave), fname, h5args=h5args)
		if mtyp is not None:
			save_model_cleaner(fname, mtyp)
	except Exception as e:
		if print_func is not None:
			print_func(str(e))

def async_save_model(model, fname, sub_module=False, print_func=print, mtyp=None, h5args=h5modelwargs, para_lock=None, log_success=None):

	def _worker(model, fname, sub_module=False, print_func=print, mtyp=None, para_lock=None, log_success=None):

		success = True
		_msave = model.module if sub_module else model
		try:
			if para_lock is None:
				h5save(mp_func(_msave), fname, h5args=h5args)
				if mtyp is not None:
					save_model_cleaner(fname, mtyp)
			else:
				with para_lock:
					h5save(mp_func(_msave), fname, h5args=h5args)
					if mtyp is not None:
						save_model_cleaner(fname, mtyp)
		except Exception as e:
			if print_func is not None:
				print_func(str(e))
			success = False
		if success and (print_func is not None) and (log_success is not None):
			print_func(str(log_success))

	Thread(target=_worker, args=(model, fname, sub_module, print_func, mtyp, para_lock, log_success)).start()

def save_states(state_dict, fname, print_func=print, mtyp=None):

	try:
		torch.save(state_dict, fname)
		if mtyp is not None:
			save_model_cleaner(fname, mtyp)
	except Exception as e:
		if print_func is not None:
			print_func(str(e))
