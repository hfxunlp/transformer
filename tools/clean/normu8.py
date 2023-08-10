#encoding: utf-8

import sys

from utils.fmt.base import sys_open
from utils.fmt.u8 import norm_u8_byte, uni_normer

def handle(srcf, rsf, uni_normer=uni_normer):

	ens="\n".encode("utf-8")
	with sys_open(srcf, "rb") as frd, sys_open(rsf, "wb") as fwrt:
		for line in frd:
			tmp = line.strip()
			if tmp:
				fwrt.write(norm_u8_byte(tmp))
			fwrt.write(ens)

if __name__ == "__main__":
	handle(*sys.argv[1:])
