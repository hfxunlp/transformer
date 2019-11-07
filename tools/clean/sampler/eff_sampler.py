#encoding: utf-8

# usage:
#	python tools/clean/sampler/eff_sampler.py srcf1 ... srcfn tgtf1 ... tgtfn keep_ratio

import sys
from random import random
from random import seed as rpyseed

def handle(srcfl, tgtfl, ratio):

	sfl = [open(srcf, "rb") for srcf in srcfl]
	tfl = [open(tgtf, "wb") for tgtf in tgtfl]

	ens = "\n".encode("utf-8")

	nkeep = ntotal = 0

	for srcl in zip(*sfl):

		if random() <=ratio:
			tmp = [tl.strip().decode("utf-8") for tl in srcl]
			for line, wrtf in zip(tmp, tfl):
				wrtf.write(line.encode("utf-8"))
				wrtf.write(ens)
				nkeep += 1
		ntotal += 1

	for f in sfl:
		f.close()
	for f in tfl:
		f.close()

	print("%d in %d data keeped with ratio %.2f" % (nkeep, ntotal, float(nkeep) / float(ntotal) * 100.0 if ntotal > 0 else 0.0))

if __name__ == "__main__":
	rpyseed(666666)
	spind = len(sys.argv) // 2
	handle(sys.argv[1:spind], sys.argv[spind:-1], float(sys.argv[-1]))
