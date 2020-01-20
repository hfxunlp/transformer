#encoding: utf-8

# usage: python tools/check/rank.py srcf tgtf rankf rssf rstf threshold

import sys

from utils.fmt.base import clean_str

def handle(srcf, tgtf, rankf, rssf, rstf, threshold):

	with open(srcf, "rb") as frs, open(tgtf, "rb") as frt, open(rankf, "rb") as fs, open(rssf, "wb") as fws, open(rstf, "wb") as fwt:

		ndata = nkeep = 0

		ens = "\n".encode("utf-8")

		for srcl, tgtl, score in zip(frs, frt, fs):
			src, tgt, s = srcl.strip(), tgtl.strip(), score.strip()
			if src and tgt and s:
				src, tgt, s = clean_str(src.decode("utf-8")), clean_str(tgt.decode("utf-8")), float(s.decode("utf-8"))
				if s <= threshold:
					fws.write(src.encode("utf-8"))
					fws.write(ens)
					fwt.write(tgt.encode("utf-8"))
					fwt.write(ens)
					nkeep += 1
				ndata += 1

		print("%d in %d data keeped with ratio %.2f" % (nkeep, ndata, float(nkeep) / float(ndata) * 100.0 if ndata > 0 else 0.0))

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], float(sys.argv[6]))
