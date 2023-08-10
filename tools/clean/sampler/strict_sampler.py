#encoding: utf-8

# usage:
#	python tools/clean/sampler/strict_sampler.py srcf1 ... srcfn tgtf1 ... tgtfn keep_ratio

import sys
from random import seed as rpyseed, shuffle

from utils.fmt.base import FileList, sys_open

def handle(srcfl, tgtfl, ratio):

	rs = []
	with FileList(srcfl, "rb") as fl:
		for srcl in zip(*fl):
			tmp = [tl.strip().decode("utf-8").encode("utf-8") for tl in srcl]
			rs.append(tmp)

	shuffle(rs)
	ntotal = len(rs)
	nkeep = int(ntotal * ratio)
	rs = zip(*rs[:nkeep])

	ens = "\n".encode("utf-8")
	for data, tgtf in zip(rs, tgtfl):
		with sys_open(tgtf, "wb") as f:
			# following 3 lines for memory
			#for line in data:
				#f.write(line)
				#f.write(ens)
			# use following lines for efficiency
			f.write(ens.join(data))
			f.write(ens)

	print("%d in %d data keeped with ratio %.2f" % (nkeep, ntotal, float(nkeep) / float(ntotal) * 100.0 if ntotal > 0 else 0.0))

if __name__ == "__main__":
	rpyseed(666666)
	spind = len(sys.argv) // 2
	handle(sys.argv[1:spind], sys.argv[spind:-1], float(sys.argv[-1]))
