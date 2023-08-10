#encoding: utf-8

import sys
import torch
from random import seed as rpyseed, shuffle

from utils.h5serial import h5File
from utils.tqdm import tqdm

from cnfg.ihyp import tqdm_mininterval

def handle(h5f, bsize, shuf=True):

	with h5File(h5f, "r") as td:
		tl = [(str(nsent), str(_curd),) for nsent, ndata in zip(td["nsent"][()].tolist(), td["ndata"][()].tolist()) for _curd in range(ndata)]
		if shuf:
			shuffle(tl)

		tgt_grp = td["tgt"]
		ntoken = 0
		nstep = 0
		for nsent, i_d in tqdm(tl, mininterval=tqdm_mininterval):
			seq_batch = torch.from_numpy(tgt_grp[nsent][i_d][()])
			ot = seq_batch.narrow(-1, 1, seq_batch.size(-1) - 1)
			ntoken += ot.ne(0).int().sum().item()
			if ntoken >= bsize:
				nstep += 1
				ntoken = 0

	return nstep

if __name__ == "__main__":
	rpyseed(666666)
	print(handle(sys.argv[1], int(sys.argv[2])))
