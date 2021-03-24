#encoding: utf-8

import sys

from utils.fmt.base import ldvocab_list, all_in

def handle(srcfs, srcft, tgtfs, tgtft, vcbft):

	ens = "\n".encode("utf-8")

	vcbt, nvt = ldvocab_list(vcbft)
	vcbt = set(vcbt)

	with open(srcfs, "rb") as fs, open(srcft, "rb") as ft, open(tgtfs, "wb") as fsw, open(tgtft, "wb") as ftw:
		total = keep = 0
		for ls, lt in zip(fs, ft):
			ls, lt = ls.strip(), lt.strip()
			if ls and lt:
				ls, lt = ls.decode("utf-8"), lt.decode("utf-8")
				if all_in(lt.split(), vcbt):
					fsw.write(ls.encode("utf-8"))
					fsw.write(ens)
					ftw.write(lt.encode("utf-8"))
					ftw.write(ens)
					keep += 1
				total += 1
		print("%d in %d data keeped with ratio %.2f" % (keep, total, float(keep) / float(total) * 100.0 if total > 0 else 0.0))

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
