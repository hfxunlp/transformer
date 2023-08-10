#encoding: utf-8

from torch.nn import ModuleList

from modules.attn.retr import SelfAttn

def share_retrattn_cache(netin):

	rel_cache_d = {}
	for net in netin.modules():
		if isinstance(net, ModuleList):
			_cache = None
			for layer in net.modules():
				if isinstance(layer, SelfAttn):
					if layer.csum is not None:
						if _cache is None:
							_cache = layer.csum
						else:
							layer.register_buffer("csum", _cache, persistent=False)

	return netin
