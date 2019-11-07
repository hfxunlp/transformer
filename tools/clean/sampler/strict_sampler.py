#encoding: utf-8

# usage:
#	python tools/clean/sampler/strict_sampler.py srcf1 ... srcfn tgtf1 ... tgtfn keep_ratio

import sys
from random import shuffle
from random import seed as rpyseed

def handle(srcfl, tgtfl, ratio):

	fl = [open(srcf, "rb") for srcf in srcfl]

	rs = []
	for srcl in zip(*fl):
		tmp = [tl.strip().decode("utf-8") for tl in srcl]
		rs.append(tmp)

	for f in fl:
		f.close()

	shuffle(rs)
	ntotal = len(rs)
	nkeep = int(ntotal * ratio)
	rs = zip(*rs[:nkeep])

	ens = "\n".encode("utf-8")
	for data, tgtf in zip(rs, tgtfl):
		with open(tgtf, "wb") as f:
			# following 3 lines for memory
			#for line in data:
				#f.write(line.encode("utf-8"))
				#f.write(ens)
			# use following lines for efficiency
			f.write("\n".join(data).encode("utf-8"))
			f.write(ens)

	print("%d in %d data keeped with ratio %.2f" % (nkeep, ntotal, float(nkeep) / float(ntotal) * 100.0 if ntotal > 0 else 0.0))

if __name__ == "__main__":
	rpyseed(666666)
	spind = len(sys.argv) // 2
	handle(sys.argv[1:spind], sys.argv[spind:-1], float(sys.argv[-1]))
