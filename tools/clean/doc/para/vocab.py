#encoding: utf-8

import sys

from utils.fmt.base import ldvocab_list
from utils.fmt.doc.base import legal_vocab

# vratio: percentages of vocabulary size of retrieved words of least frequencies
# dratio: a datum will be dropped who contains high frequency words less than this ratio

def handle(srcfs, srcft, tgtfs, tgtft, vcbfs, vcbft, vratio, dratio=None):

	_dratio = vratio if dratio is None else dratio

	ens = "\n\n".encode("utf-8")

	vcbs, nvs = ldvocab_list(vcbfs)
	vcbt, nvt = ldvocab_list(vcbft)
	ilgs = set(vcbs[int(float(nvs) * (1.0 - vratio)):])
	ilgt = set(vcbt[int(float(nvt) * (1.0 - vratio)):])

	cache_s, cache_t = [], []
	with open(srcfs, "rb") as fs, open(srcft, "rb") as ft, open(tgtfs, "wb") as fsw, open(tgtft, "wb") as ftw:
		total = keep = 0
		for ls, lt in zip(fs, ft):
			ls, lt = ls.strip(), lt.strip()
			if ls and lt:
				cache_s.append(ls.decode("utf-8"))
				cache_t.append(lt.decode("utf-8"))
			elif cache_s and cache_t:
				if legal_vocab(cache_s, ilgs, _dratio) and legal_vocab(cache_t, ilgt, _dratio):
					fsw.write("\n".join(cache_s).encode("utf-8"))
					fsw.write(ens)
					ftw.write("\n".join(cache_t).encode("utf-8"))
					ftw.write(ens)
					keep += 1
				total += 1
				cache_s, cache_t = [], []
		if cache_s and cache_t:
			if legal_vocab(cache_s, ilgs, _dratio) and legal_vocab(cache_t, ilgt, _dratio):
				fsw.write("\n".join(cache_s).encode("utf-8"))
				fsw.write(ens)
				ftw.write("\n".join(cache_t).encode("utf-8"))
				ftw.write(ens)
				keep += 1
			total += 1
			cache_s, cache_t = [], []
		print("%d in %d data keeped with ratio %.2f" % (keep, total, float(keep) / float(total) * 100.0 if total > 0 else 0.0))

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], float(sys.argv[7]))
