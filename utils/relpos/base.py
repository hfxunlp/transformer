#encoding: utf-8

from torch.nn import ModuleList

from modules.base import CrossAttn, MultiHeadAttn, SelfAttn

def share_rel_pos_cache(netin, share_emb=False):

	rel_cache_d = {}
	rel_map_cache_d = {}
	for net in netin.modules():
		if isinstance(net, ModuleList):
			base_nets = {}
			for layer in net.modules():
				if isinstance(layer, (SelfAttn, MultiHeadAttn, CrossAttn,)):
					if hasattr(layer, "rel_pemb") and (layer.rel_pemb is not None):
						_key_rel_pos_map = None if layer.rel_pos_map is None else layer.rel_pos_map.size()
						_key = (layer.clamp_min, layer.clamp_max, layer.rel_shift, _key_rel_pos_map,)
						if _key in base_nets:
							layer.ref_rel_posm = base_nets[_key]
							if share_emb:
								layer.rel_pemb = base_nets[_key].rel_pemb
						else:
							base_nets[_key] = layer
						if _key_rel_pos_map is not None:
							if _key in rel_map_cache_d:
								layer.register_buffer("rel_pos_map", rel_map_cache_d[_key], persistent=False)
							else:
								rel_map_cache_d[_key] = layer.rel_pos_map
						_key = (layer.clamp_min, layer.clamp_max, layer.rel_shift, _key_rel_pos_map, layer.rel_pos.size(),)
						if _key in rel_cache_d:
							layer.register_buffer("rel_pos", rel_cache_d[_key], persistent=False)
						else:
							rel_cache_d[_key] = layer.rel_pos

	return netin
