#encoding: utf-8

import sys

def handle(srcfs, srtsf, srttf, tgtf):

	def clean(lin):
		rs = []
		for lu in lin.split():
			if lu:
				rs.append(lu)
		return " ".join(rs), len(rs)

	data = {}

	with open(srtsf, "rb") as fs, open(srttf, "rb") as ft:
		for sl, tl in zip(fs, ft):
			_sl, _tl = sl.strip(), tl.strip()
			if _sl and _tl:
				_sl, _ls = clean(_sl.decode("utf-8"))
				_tl, _lt = clean(_tl.decode("utf-8"))
			data[_sl] = _tl

	ens = "\n".encode("utf-8")

	with open(srcfs, "rb") as fs, open(tgtf, "wb") as ft:
		for line in fs:
			tmp = line.strip()
			if tmp:
				tmp, _ = clean(tmp.decode("utf-8"))
				tmp = data.get(tmp, "")
				ft.write(tmp.encode("utf-8"))
			ft.write(ens)

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
