#encoding: utf-8

''' usage:
	python remove_layers.py $src.t7 $rs.t7 enc/dec layers...
'''

import sys

import torch
from torch.nn import ModuleList

from transformer.NMT import NMT

from utils import *

import h5py
import cnfg

def handle(srcf, rsf, typ, rlist):

	td = h5py.File(cnfg.dev_data, "r")
	nwordi = int(td["nwordi"][:][0])
	nwordt = int(td["nwordt"][:][0])
	td.close()

	_tmpm = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.share_emb, cnfg.nhead, cnfg.cache_len, cnfg.attn_hsize, cnfg.norm_output, cnfg.bindDecoderEmb, cnfg.forbidden_indexes)
	_tmpm = load_model_cpu(srcf, _tmpm)
	if typ == "enc":
		_tmpm.enc.nets = ModuleList(remove_layers(list(_tmpm.enc.nets), rlist))
	elif typ == "dec":
		_tmpm.dec.nets = ModuleList(remove_layers(list(_tmpm.dec.nets), rlist))

	save_model(_tmpm, rsf, False)

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3], [int(_t) for _t in sys.argv[4:]])
