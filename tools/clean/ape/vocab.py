#encoding: utf-8

import sys

from utils.fmt.base import ldvocab_list, legal_vocab

# vratio: percentages of vocabulary size of retrieved words of least frequencies
# dratio: a datum will be dropped who contains high frequency words less than this ratio

def handle(srcfs, srcfm, srcft, tgtfs, tgtfm, tgtft, vcbfs, vcbft, vratio, dratio=None):

	_dratio = vratio if dratio is None else dratio

	ens = "\n".encode("utf-8")

	vcbs, nvs = ldvocab_list(vcbfs)
	vcbt, nvt = ldvocab_list(vcbft)
	ilgs = set(vcbs[int(float(nvs) * (1.0 - vratio)):])
	ilgt = set(vcbt[int(float(nvt) * (1.0 - vratio)):])

	with open(srcfs, "rb") as fs, open(srcfm, "rb") as fm, open(srcft, "rb") as ft, open(tgtfs, "wb") as fsw, open(tgtfm, "wb") as fmw, open(tgtft, "wb") as ftw:
		total = keep = 0
		for ls, lm, lt in zip(fs, fm, ft):
			ls, lm, lt = ls.strip(), lm.strip(), lt.strip()
			if ls and lm and lt:
				ls, lm, lt = ls.decode("utf-8"), lm.decode("utf-8"), lt.decode("utf-8")
				if legal_vocab(ls, ilgs, _dratio) and legal_vocab(lt, ilgt, _dratio):
					fsw.write(ls.encode("utf-8"))
					fsw.write(ens)
					fmw.write(lm.encode("utf-8"))
					fmw.write(ens)
					ftw.write(lt.encode("utf-8"))
					ftw.write(ens)
					keep += 1
				total += 1
		print("%d in %d data keeped with ratio %.2f" % (keep, total, float(keep) / float(total) * 100.0 if total > 0 else 0.0))

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], float(sys.argv[9]))
