#encoding: utf-8

from modules.base import MultiHeadAttn, SelfAttn

def share_rel_pos_cache_core(netsin, retrieve_func=None):# lambda m: m.attn, m.self_attn for encoder, decoder layer

	_base_net = netsin[0]
	_base_sattn = None
	_base_cache = None
	if retrieve_func is None:
		# dangerous in case decoder layers using MultiHeadAttn for both self attention and cross attention
		for _m in _base_net.modules():
			if isinstance(_m, (SelfAttn, MultiHeadAttn,)):
				_base_sattn = _m
				if _base_sattn.rel_pemb is not None:
					_base_cache = _base_sattn.rel_pos
					break
	else:
		_base_sattn = retrieve_func(_base_net)
		if _base_sattn.rel_pemb is not None:
			_base_cache = _base_sattn.rel_pos

	# assign ref module, share cache
	if _base_cache is not None:
		for _net in netsin[1:]:
			_sel_net = None
			if retrieve_func is None:
				for _m in _base_net.modules():
					if isinstance(_m, (SelfAttn, MultiHeadAttn,)) and (_m.rel_pemb is not None):
						_sel_net = _m
						break
			else:
				_sel_net = retrieve_func(_net)
				if _sel_net.rel_pemb is None:
					_sel_net = None
			if _sel_net is not None:
				_sel_net.ref_rel_posm, _sel_net.rel_pos = _base_sattn, _base_cache

	return netsin

def share_rel_pos_cache_enc(netsin):

	return share_rel_pos_cache_core(netsin, retrieve_func=lambda m: m.attn)

def share_rel_pos_cache_dec(netsin):

	return share_rel_pos_cache_core(netsin, retrieve_func=lambda m: m.self_attn)

def share_rel_pos_cache(encnets, decnets):

	if decnets[0].self_attn.rel_pemb is not None:
		decnets[0].self_attn.rel_pos = encnets[0].attn.rel_pos

	return share_rel_pos_cache_enc(encnets), share_rel_pos_cache_dec(decnets)
