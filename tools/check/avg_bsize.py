#encoding: utf-8

import sys
import torch
from random import seed as rpyseed, shuffle

from utils.h5serial import h5File
from utils.tqdm import tqdm

from cnfg.ihyp import tqdm_mininterval

def handle(h5f, bsize, shuf=True):

	ntoken = 0
	rsl = []
	with h5File(h5f, "r") as td:
		ntest = td["ndata"][()].item()
		tl = list(range(ntest))
		if shuf:
			shuffle(tl)

		tgt_grp = td["tgt"]
		for tid in tqdm(tl, mininterval=tqdm_mininterval):
			seq_batch = torch.from_numpy(tgt_grp[str(tid)][()])
			ot = seq_batch.narrow(-1, 1, seq_batch.size(-1) - 1)
			ntoken += ot.ne(0).int().sum().item()
			if ntoken >= bsize:
				rsl.append(ntoken)
				ntoken = 0

	return sum(rsl)/float(len(rsl))

if __name__ == "__main__":
	rpyseed(666666)
	print("%.2f" % (handle(sys.argv[1], int(sys.argv[2])),))
