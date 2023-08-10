#encoding: utf-8

from torch import nn

def copy_plm_parameter(src, plm_parameters, keys, func=None, func_args=None, func_kwargs=None, print_func=print):

	_tgt = None
	if isinstance(keys, str):
		_p_k = keys
		if keys in plm_parameters:
			_tgt = plm_parameters[keys]
	else:
		_p_k = str(keys)
		if all(_ in plm_parameters for _ in keys):
			_tgt = [plm_parameters[_] for _ in keys]
	if _tgt is not None:
		if func is not None:
			_tgt = func(_tgt, *([] if func_args is None else func_args), **({} if func_kwargs is None else func_kwargs))
		_src = src
		_s_size, _t_size = _src.size(), _tgt.size()
		if len(_s_size) == len(_t_size):
			_mdl = []
			for _i, (_s, _t,) in enumerate(zip(_s_size, _t_size)):
				if _s > _t:
					_src = _src.narrow(_i, 0, _t)
					_mdl.append(_i)
				elif _s < _t:
					_tgt = _tgt.narrow(_i, 0, _s)
					_mdl.append(_i)
			_src.copy_(_tgt)
			if _mdl and (print_func is not None):
				print_func("size mismatch for %s at dimension(s) %s" % (_p_k, ",".join([str(_) for _ in _mdl]),))
				print_func(_s_size, _t_size)
		elif print_func is not None:
			print_func("dimension mismatch for %s" % _p_k)
			print_func(_s_size, _t_size)
	elif print_func is not None:
		print_func("%s does not exist" % _p_k)

def set_ln_ieps(netin, ieps):

	for net in netin.modules():
		if isinstance(net, nn.LayerNorm) and hasattr(net, "eps") and (net.eps != ieps):
			net.eps = ieps

	return netin
