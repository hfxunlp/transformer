#encoding: utf-8

import sys

from utils.fmt.base import sys_open
from utils.fmt.vocab.token import init_vocab, ldvocab

def handle(vcbf, srcfl, fvocab_task, rsf, minfreq=False, vsize=False):

	vcb, nwords = ldvocab(vcbf, minf=minfreq, omit_vsize=vsize, vanilla=False)
	vcbtask, nwordtask = ldvocab(fvocab_task, minf=False, omit_vsize=False, vanilla=True)

	fvcb = {}

	for srcf, tgtf in zip(srcfl[0::2], srcfl[1::2]):
		with sys_open(srcf, "rb") as fsrc, sys_open(tgtf, "rb") as ftgt:
			for lsrc, ltgt in zip(fsrc, ftgt):
				tsrc, ttgt = lsrc.strip(), ltgt.strip()
				if tsrc and ttgt:
					task = vcbtask[tsrc.decode("utf-8").split()[0]]
					if task not in fvcb:
						fvcb[task] = set(init_vocab.keys())
					wset = fvcb[task]
					for token in ttgt.decode("utf-8").split():
						if token and (token not in wset):
							wset.add(token)

	rsl = []
	for i in range(nwordtask):
		wset = fvcb[i]
		tmp = []
		for wd, ind in vcb.items():
			if wd not in wset:
				tmp.append(ind)
		rsl.append(tmp)

	with sys_open(rsf, "wb") as f:
		f.write("#encoding: utf-8\n\nfbl = ".encode("utf-8"))
		f.write(repr(rsl).encode("utf-8"))
		f.write("\n".encode("utf-8"))

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2:-2], sys.argv[-2], sys.argv[-1])
