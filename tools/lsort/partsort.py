#encoding: utf-8

import sys
from os import path

from utils.fmt.base import clean_list_len, iter_dict_sort, dict_insert_list

def handle(srcfs, srcft, tgtd, max_len=256, cache_token=268435456):

	def save_cache(cache, srcf, tgtf):

		ens = "\n".encode("utf-8")

		with open(srcf, "wb") as fs, open(tgtf, "wb") as ft:
			for tmp in iter_dict_sort(cache):
				ls, lt = zip(*tmp)
				fs.write("\n".join(ls).encode("utf-8"))
				fs.write(ens)
				ft.write("\n".join(lt).encode("utf-8"))
				ft.write(ens)

	_max_len = max(1, max_len - 2)

	data = {}

	mem_token = curf = 0

	with open(srcfs, "rb") as fs, open(srcft, "rb") as ft:
		for ls, lt in zip(fs, ft):
			ls, lt = ls.strip(), lt.strip()
			if ls and lt:
				ls, slen = clean_list_len(ls.decode("utf-8").split())
				lt, tlen = clean_list_len(lt.decode("utf-8").split())
				if (slen <= _max_len) and (tlen <= _max_len):
					lgth = slen + tlen
					data = dict_insert_list(data, (ls, lt,), lgth, tlen)
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
