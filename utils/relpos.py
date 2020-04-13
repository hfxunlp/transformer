#encoding: utf-8

from torch.nn import ModuleList
from modules.base import MultiHeadAttn, SelfAttn

def share_rel_pos_cache(netin):

	rel_cache_d = {}
	for net in netin.modules():
		if isinstance(net, ModuleList):
			_base_net = None
			for layer in net.modules():
				if isinstance(layer, (SelfAttn, MultiHeadAttn,)):
					if layer.rel_pemb is not None:
						if _base_net is None:
							_base_net = layer
						else:
							layer.ref_rel_posm = _base_net
						_rel_c_size = layer.rel_pos.size()
						if _rel_c_size in rel_cache_d:
							layer.rel_pos = rel_cache_d[_rel_c_size]
						else:
							rel_cache_d[_rel_c_size] = layer.rel_pos

	return netin
