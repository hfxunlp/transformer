#encoding: utf-8

import sys

from utils.fmt.base import clean_str

def handle(srcfs, srtsf_base, srttf_base, srtsf, srttf, tgtf):

	data = {}

	with open(srtsf_base, "rb") as fs, open(srttf_base, "rb") as ft:
		for sl, tl in zip(fs, ft):
			_sl, _tl = sl.strip(), tl.strip()
			if _sl and _tl:
				_sl = clean_str(_sl.decode("utf-8"))
				_tl = clean_str(_tl.decode("utf-8"))
				data[_sl] = _tl
	with open(srtsf, "rb") as fs, open(srttf, "rb") as ft:
		for sl, tl in zip(fs, ft):
			_sl, _tl = sl.strip(), tl.strip()
			if _sl and _tl:
				_sl = clean_str(_sl.decode("utf-8"))
				_tl = clean_str(_tl.decode("utf-8"))
				data[_sl] = _tl

	ens = "\n".encode("utf-8")

	with open(srcfs, "rb") as fs, open(tgtf, "wb") as ft:
		for line in fs:
			tmp = line.strip()
			if tmp:
				tmp = clean_str(tmp.decode("utf-8"))
				tmp = data.get(tmp, "")
				ft.write(tmp.encode("utf-8"))
			ft.write(ens)

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
