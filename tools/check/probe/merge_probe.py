#encoding: utf-8

import sys

from transformer.NMT import NMT as NMTBase
from transformer.Probe.NMT import NMT
from utils.h5serial import h5File
from utils.init.base import init_model_params
from utils.io import load_model_cpu, save_model

import cnfg.probe as cnfg
from cnfg.ihyp import *

def handle(cnfg, srcmtf, decf, rsf):

	with h5File(cnfg.dev_data, "r") as tdf:
		nwordi, nwordt = tdf["nword"][()].tolist()

	mymodel = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.act_drop, cnfg.share_emb, cnfg.nhead, cache_len_default, cnfg.attn_hsize, cnfg.norm_output, cnfg.bindDecoderEmb, cnfg.forbidden_indexes, cnfg.num_layer_fwd)
	init_model_params(mymodel)
	_tmpm = NMTBase(cnfg.isize, nwordi, nwordt, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.act_drop, cnfg.share_emb, cnfg.nhead, cache_len_default, cnfg.attn_hsize, cnfg.norm_output, cnfg.bindDecoderEmb, cnfg.forbidden_indexes)
	_tmpm = init_model_params(_tmpm)
	_tmpm = load_model_cpu(srcmtf, _tmpm)
	mymodel.load_base(_tmpm)
	mymodel.dec = load_model_cpu(decf, mymodel.dec)
	if cnfg.share_emb:
		mymodel.dec.wemb.weight = _tmpm.enc.wemb.weight
	if cnfg.bindDecoderEmb:
		mymodel.dec.classifier.weight = mymodel.dec.wemb.weight
	_tmpm = None

	save_model(mymodel, rsf, sub_module=False, h5args=h5zipargs)

if __name__ == "__main__":
	handle(cnfg, sys.argv[1], sys.argv[2], sys.argv[3])
