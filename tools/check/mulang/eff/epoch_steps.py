#encoding: utf-8

import sys
import torch
from random import seed as rpyseed, shuffle

from utils.h5serial import h5File
from utils.tqdm import tqdm

from cnfg.ihyp import tqdm_mininterval

def handle(h5f, bsize, shuf=True):

	with h5File(h5f, "r") as td:
		ntest = td["ndata"][()].tolist()
		tl = [(i, str(_task),) for _nd, _task in zip(ntest, td["taskorder"][()].tolist()) for i in range(_nd)]
		if shuf:
			shuffle(tl)

		ntoken = 0
		nstep = 0
		for tid, taskid in tqdm(tl, mininterval=tqdm_mininterval):
			seq_batch = torch.from_numpy(td[taskid]["tgt"][str(tid)][()])
			ot = seq_batch.narrow(-1, 1, seq_batch.size(-1) - 1)
			ntoken += ot.ne(0).int().sum().item()
			if ntoken >= bsize:
				nstep += 1
				ntoken = 0

	return nstep

if __name__ == "__main__":
	rpyseed(666666)
	print(handle(sys.argv[1], int(sys.argv[2])))
