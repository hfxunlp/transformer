#encoding: utf-8

# usage:
#	python tools/clean/sampler/eff_sampler.py srcf1 ... srcfn tgtf1 ... tgtfn keep_ratio

import sys
from random import random, seed as rpyseed

from utils.fmt.base import FileList

def handle(srcfl, tgtfl, ratio):

	ens = "\n".encode("utf-8")
	nkeep = ntotal = 0

	with FileList(srcfl, "rb") as sfl, FileList(tgtfl, "rb") as tfl:
		for srcl in zip(*sfl):
			if random() <=ratio:
				tmp = [tl.strip().decode("utf-8").encode("utf-8") for tl in srcl]
				for line, wrtf in zip(tmp, tfl):
					wrtf.write(line)
					wrtf.write(ens)
					nkeep += 1
			ntotal += 1

	print("%d in %d data keeped with ratio %.2f" % (nkeep, ntotal, float(nkeep) / float(ntotal) * 100.0 if ntotal > 0 else 0.0))

if __name__ == "__main__":
	rpyseed(666666)
	spind = len(sys.argv) // 2
	handle(sys.argv[1:spind], sys.argv[spind:-1], float(sys.argv[-1]))
