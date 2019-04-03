#encoding: utf-8

import sys
from os import path

def handle(srcfs, srcft, tgtd, max_len=256, cache_token=268435456):

	def clean(lin):
		rs = []
		for lu in lin:
			if lu:
				rs.append(lu)
		return " ".join(rs), len(rs)

	def save_cache(cache, srcf, tgtf):

		length = list(cache.keys())
		length.sort()

		ens = "\n".encode("utf-8")

		with open(srcf, "wb") as fs, open(tgtf, "wb") as ft:
			for lgth in length:
				lg = list(cache[lgth].keys())
				lg.sort()
				for lu in lg:
					ls, lt = zip(*cache[lgth][lu])
					fs.write("\n".join(ls).encode("utf-8"))
					fs.write(ens)
					ft.write("\n".join(lt).encode("utf-8"))
					ft.write(ens)

	_max_len = max(1, max_len - 2)

	data = {}

	mem_token = 0
	curf = 0
	with open(srcfs, "rb") as fs, open(srcft, "rb") as ft:
		for ls, lt in zip(fs, ft):
			ls, lt = ls.strip(), lt.strip()
			if ls and lt:
				ls, slen = clean(ls.decode("utf-8").split())
				lt, tlen = clean(lt.decode("utf-8").split())
				if (slen <= _max_len) and (tlen <= _max_len):
					lgth = slen + tlen
					if lgth not in data:
						data[lgth] = {tlen: [(ls, lt)]}
					else:
						if tlen in data[lgth]:
							data[lgth][tlen].append((ls, lt))
						else:
							data[lgth][tlen] = [(ls, lt)]
					mem_token += lgth
					if mem_token > cache_token:
						_curfid = str(curf)
						save_cache(data, path.join(tgtd, _curfid + ".src"), path.join(tgtd, _curfid + ".tgt"))
						data = {}
						mem_token = 0
						curf += 1
	if data:
		_curfid = str(curf)
		save_cache(data, path.join(tgtd, _curfid + ".src"), path.join(tgtd, _curfid + ".tgt"))

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
