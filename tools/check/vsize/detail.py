#encoding: utf-8

import sys

from utils.fmt.base import clean_list_iter

def collect(fl):

	vcb = set()
	for srcf in fl:
		with open(srcf, "rb") as f:
			for line in f:
				tmp = line.strip()
				if tmp:
					for token in clean_list_iter(tmp.decode("utf-8").split()):
						if not token in vcb:
							vcb.add(token)

	return vcb

def handle(srcfl, tgtfl):

	src_vcb, tgt_vcb = collect(srcfl), collect(tgtfl)

	print("src/tgt vcb: %d, %d, shared token: %d" % (len(src_vcb), len(tgt_vcb), len(src_vcb&tgt_vcb),))

if __name__ == "__main__":
	sep_index = (len(sys.argv) + 1) // 2
	handle(sys.argv[1:sep_index], sys.argv[sep_index:])
