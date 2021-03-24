#encoding: utf-8

import sys

import h5py

from utils.base import *
from utils.init import init_model_params

import cnfg.probe as cnfg
from cnfg.ihyp import *

from transformer.NMT import NMT as NMTBase
from transformer.Probe.NMT import NMT

def handle(cnfg, srcmtf, decf, rsf):

	tdf = h5py.File(cnfg.dev_data, "r")
	nwordi, nwordt = tdf["nword"][:].tolist()
	tdf.close()

	mymodel = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.share_emb, cnfg.nhead, cache_len_default, cnfg.attn_hsize, cnfg.norm_output, cnfg.bindDecoderEmb, cnfg.forbidden_indexes, cnfg.num_layer_fwd)
	init_model_params(mymodel)
	_tmpm = NMTBase(cnfg.isize, nwordi, nwordt, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.share_emb, cnfg.nhead, cache_len_default, cnfg.attn_hsize, cnfg.norm_output, cnfg.bindDecoderEmb, cnfg.forbidden_indexes)
	_tmpm = init_model_params(_tmpm)
	_tmpm = load_model_cpu(srcmtf, _tmpm)
	mymodel.load_base(_tmpm)
	mymodel.dec = load_model_cpu(decf, mymodel.dec)
	if cnfg.share_emb:
		mymodel.dec.wemb.weight = _tmpm.enc.wemb.weight
	if cnfg.bindDecoderEmb:
		mymodel.dec.classifier.weight = mymodel.dec.wemb.weight
	_tmpm = None

	save_model(mymodel, rsf, sub_module=False, logger=None, h5args=h5zipargs)

if __name__ == "__main__":
	handle(cnfg, sys.argv[1], sys.argv[2], sys.argv[3])
