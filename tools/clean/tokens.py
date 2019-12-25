#encoding: utf-8

import sys

from utils.fmt.base import clean_liststr_lentok

def handle(srcfs, srcft, tgtfs, tgtft, maxlen=256):

	ens = "\n".encode("utf-8")

	with open(srcfs, "rb") as fs, open(srcft, "rb") as ft, open(tgtfs, "wb") as fsw, open(tgtft, "wb") as ftw:
		total = keep = 0
		for ls, lt in zip(fs, ft):
			ls, lt = ls.strip(), lt.strip()
			if ls and lt:
				ls, lt = ls.decode("utf-8"), lt.decode("utf-8")
				ls, lens = clean_liststr_lentok(ls.split())
				lt, lent = clean_liststr_lentok(lt.split())
				if (lens <= maxlen) and (lent <= maxlen):
					fsw.write(ls.encode("utf-8"))
					fsw.write(ens)
					ftw.write(lt.encode("utf-8"))
					ftw.write(ens)
					keep += 1
				total += 1
	print("%d in %d data keeped with ratio %.2f" % (keep, total, float(keep) / float(total) * 100.0 ))

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
