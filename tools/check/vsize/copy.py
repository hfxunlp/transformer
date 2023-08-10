#encoding: utf-8

import sys

from utils.fmt.base import clean_list, sys_open

def handle(srcfl, tgtfl):

	nsrc = ntgt = ncopy = 0
	for srcf, tgtf in zip(srcfl, tgtfl):
		with sys_open(srcf, "rb") as fsrc, sys_open(tgtf, "rb") as ftgt:
			for srcl, tgtl in zip(fsrc, ftgt):
				srcl, tgtl = srcl.strip(), tgtl.strip()
				if srcl or tgtl:
					srcvcb, tgtvcb = clean_list(srcl.decode("utf-8").split()), clean_list(tgtl.decode("utf-8").split())
					nsrc += len(srcvcb)
					ntgt += len(tgtvcb)
					ncopy += len(set(srcvcb)&set(tgtvcb))

	print("src, tgt, copy: %d, %d, %d" % (nsrc, ntgt, ncopy,))

if __name__ == "__main__":
	sep_index = (len(sys.argv) + 1) // 2
	handle(sys.argv[1:sep_index], sys.argv[sep_index:])
