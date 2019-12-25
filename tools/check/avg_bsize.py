#encoding: utf-8

import sys

import torch
import h5py

from tqdm import tqdm
from random import shuffle
from random import seed as rpyseed

def handle(h5f, bsize, shuf=True):

	td = h5py.File(h5f, "r")
	ntest = td["ndata"][:].item()
	tl = list(range(ntest))
	if shuf:
		shuffle(tl)

	tgt_grp = td["tgt"]
	ntoken = 0
	rsl = []
	for tid in tqdm(tl):
		seq_batch = torch.from_numpy(tgt_grp[str(tid)][:])
		ot = seq_batch.narrow(-1, 1, seq_batch.size(-1) - 1)
		ntoken += ot.ne(0).int().sum().item()
		if ntoken >= bsize:
			rsl.append(ntoken)
			ntoken = 0

	td.close()

	return sum(rsl)/float(len(rsl))

if __name__ == "__main__":
	rpyseed(666666)
	print("%.2f" % (handle(sys.argv[1], int(sys.argv[2])),))
