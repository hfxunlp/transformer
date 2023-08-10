#encoding: utf-8

""" usage:
	python stat.py $topk_file.h5 $src_file.h5
"""

import sys
import torch

from utils.h5serial import h5File
from utils.torch.comp import torch_inference_mode
from utils.tqdm import tqdm

from cnfg.ihyp import *
from cnfg.vocab.base import pad_id

def handle(srcf, ref):

	with h5File(srcf, "r") as fs, h5File(ref, "r") as fr, torch_inference_mode():
		p_grp, ind_grp, tgt_grp = fs["p"], fs["ind"], fr["tgt"]
		ndata = fs["ndata"][()].item()
		nword = 0
		ntopk = torch.from_numpy(p_grp["0"][()]).size(-1)
		cum_p = torch.zeros(ntopk, dtype=torch.double)
		m_ind = torch.zeros(ntopk, dtype=torch.long)
		for i in tqdm(range(ndata), mininterval=tqdm_mininterval):
			bid = str(i)
			p = torch.from_numpy(p_grp[bid][()])
			ind = torch.from_numpy(ind_grp[bid][()])
			seq_o = torch.from_numpy(tgt_grp[bid][()])
			seq_o = seq_o.narrow(1, 1, seq_o.size(1) - 1)
			#mask = seq_o.eq(pad_id)
			#p.masked_fill_(mask.unsqueeze(-1), 0.0)
			data_mask = seq_o.ne(pad_id)
			cum_p.add_(p.view(-1, ntopk).sum(0).double())
			m_ind.add_((ind.eq(seq_o.unsqueeze(-1)) & data_mask.unsqueeze(-1)).view(-1, ntopk).long().sum(0))
			nword += data_mask.int().sum().item()
	nword = float(nword) / 100.0

	return cum_p.div_(nword).cumsum(-1), m_ind.cumsum(-1).double().div_(nword)

if __name__ == "__main__":
	c_p, c_i = handle(sys.argv[1], sys.argv[2])
	print(c_p.tolist())
	print(c_i.tolist())
