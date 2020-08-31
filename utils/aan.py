#encoding: utf-8

from torch.nn import ModuleList
from modules.base import AverageAttn

def share_aan_cache(netin):

	rel_cache_d = {}
	for net in netin.modules():
		if isinstance(net, ModuleList):
			_cache = None
			for layer in net.modules():
				if isinstance(layer, AverageAttn):
					if _cache is None:
						_cache = layer.w
					else:
						layer.w = _cache

	return netin
