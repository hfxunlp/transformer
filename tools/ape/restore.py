#encoding: utf-8

import sys

from utils.fmt.base import clean_str

def handle(srcfs, srcfm, srtsf, srtmf, srttf, tgtf):

	data = {}

	with open(srtsf, "rb") as fs, open(srtmf, "rb") as fm, open(srttf, "rb") as ft:
		for sl, ml, tl in zip(fs, fm, ft):
			_sl, _ml, _tl = sl.strip(), ml.strip(), tl.strip()
			if _sl and _tl:
				_sl = clean_str(_sl.decode("utf-8"))
				_ml = clean_str(_ml.decode("utf-8"))
				_tl = clean_str(_tl.decode("utf-8"))
				data[(_sl, _ml,)] = _tl

	ens = "\n".encode("utf-8")

	with open(srcfs, "rb") as fs, open(srcfm, "rb") as fm, open(tgtf, "wb") as ft:
		for sl, ml in zip(fs, fm):
			_sl, _ml = sl.strip(), ml.strip()
			if _sl:
				_sl = clean_str(_sl.decode("utf-8"))
				_ml = clean_str(_ml.decode("utf-8"))
				tmp = data.get((_sl, _ml,), "")
				ft.write(tmp.encode("utf-8"))
			ft.write(ens)

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
