#encoding: utf-8

from torch.nn import ModuleList
from modules.base import MultiHeadAttn, SelfAttn

def share_rel_pos_cache(netin):

	rel_cache_d = {}
	for net in netin.modules():
		if isinstance(net, ModuleList):
			base_nets = {}
			for layer in net.modules():
				if isinstance(layer, (SelfAttn, MultiHeadAttn,)):
					if layer.rel_pemb is not None:
						_key = (layer.clamp_min, layer.clamp_max, layer.rel_shift,)
						if _key in base_nets:
							layer.ref_rel_posm = base_nets[_key]
						else:
							base_nets[_key] = layer
						_key = (layer.clamp_min, layer.clamp_max, layer.rel_shift, layer.rel_pos.size(),)
						if _key in rel_cache_d:
							layer.rel_pos = rel_cache_d[_key]
						else:
							rel_cache_d[_key] = layer.rel_pos

	return netin
