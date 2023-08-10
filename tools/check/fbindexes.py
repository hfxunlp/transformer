#encoding: utf-8

import sys

from utils.fmt.base import sys_open
from utils.fmt.vocab.token import init_vocab, ldvocab

def handle(vcbf, srcfl, rsf, minfreq=False, vsize=False):

	vcb, nwords = ldvocab(vcbf, minfreq, vsize)

	fvcb = set(init_vocab.keys())

	for srcf in srcfl:
		with sys_open(srcf, "rb") as f:
			for line in f:
				tmp = line.strip()
				if tmp:
					for token in tmp.decode("utf-8").split():
						if token and (token not in fvcb):
							fvcb.add(token)

	rsl = []
	for wd, ind in vcb.items():
		if wd not in fvcb:
			rsl.append(ind)

	with sys_open(rsf, "wb") as f:
		f.write("#encoding: utf-8\n\nfbl = ".encode("utf-8"))
		f.write(repr(rsl).encode("utf-8"))
		f.write("\n".encode("utf-8"))

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2:-1], sys.argv[-1])
