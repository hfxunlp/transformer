#encoding: utf-8

""" this file aims at pruning source/target vocabulary of the trained model using a shared vocabulary. It depends on the model implementation, and has to be executed at the root path of the project. Usage:
	python prune_model_vocab.py path/to/common.vcb path/to/common.vcb path/to/src.vcb path/to/tgt.vcb path/to/model.h5 path/to/pruned_model.h5
"""

import sys
import torch

from transformer.NMT import NMT
from utils.fmt.vocab.base import reverse_dict
from utils.fmt.vocab.token import ldvocab
from utils.io import load_model_cpu, save_model

import cnfg.base as cnfg
from cnfg.ihyp import *
from cnfg.vocab.base import pad_id

def handle(vsrc, vtgt, src, tgt, srcm, rsm, minfreq=False, vsize=False):

	vcbi, nwordi = ldvocab(vsrc, minf=minfreq, omit_vsize=vsize, vanilla=False)
	vcbt, nwordt = ldvocab(vtgt, minf=minfreq, omit_vsize=vsize, vanilla=False)

	if src == vsrc:
		src_indices = None
	else:
		vcbw, nword = ldvocab(src, minf=minfreq, omit_vsize=vsize, vanilla=False)
		vcbw = reverse_dict(vcbw)
		src_indices = torch.as_tensor([vcbi.get(vcbw[i], pad_id) for i in range(nword)], dtype=torch.long)
	if tgt == vtgt:
		tgt_indices = None
	else:
		vcbw, nword = ldvocab(tgt, minf=minfreq, omit_vsize=vsize, vanilla=False)
		vcbw = reverse_dict(vcbw)
		tgt_indices = torch.as_tensor([vcbt.get(vcbw[i], pad_id) for i in range(nword)], dtype=torch.long)

	mymodel = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.act_drop, cnfg.share_emb, cnfg.nhead, cache_len_default, cnfg.attn_hsize, cnfg.norm_output, cnfg.bindDecoderEmb, cnfg.forbidden_indexes)
	mymodel = load_model_cpu(srcm, mymodel)
	mymodel.update_vocab(src_indices=src_indices, tgt_indices=tgt_indices)
	save_model(mymodel, rsm, sub_module=False, h5args=h5zipargs)

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
