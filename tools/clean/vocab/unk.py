#encoding: utf-8

import sys

from utils.fmt.base import sys_open

def handle(srcf, rsf):

	ens = "\n".encode("utf-8")

	with sys_open(srcf, "rb") as frd, sys_open(rsf, "wb") as fwt:
		for ls in frd:
			ls = ls.strip()
			if ls:
				ls = " ".join([tmpu for tmpu in ls.decode("utf-8").split() if tmpu and (tmpu != "<unk>")])
				fwt.write(ls.encode("utf-8"))
			fwt.write(ens)

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2])
