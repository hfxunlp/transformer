#encoding: utf-8

import torch
from torch.nn import ModuleList

from modules.plm.t5 import CrossAttn, SelfAttn

def reorder_pemb(w):

	_ = w.size(0) // 2

	return torch.cat((w.narrow(0, 0, _).flip(0), w.narrow(0, _ + 1, _ - 1),), dim=0)

def extend_rel_emb(netin):

	for net in netin.modules():
		if isinstance(net, ModuleList):
			base_nets = {}
			for layer in net.modules():
				if isinstance(layer, (SelfAttn, CrossAttn,)):
					_key = isinstance(layer, SelfAttn)
					if _key in base_nets:
						layer.ref_rel_emb = base_nets.get(_key, None)
					elif layer.rel_pemb is not None:
						base_nets[_key] = layer

	return netin
