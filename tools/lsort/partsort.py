#encoding: utf-8

import sys
from os.path import join as pjoin

from utils.fmt.base import FileList, all_le, clean_liststr_lentok, dict_insert_list, dict_insert_set, iter_dict_sort

def handle(srcfl, tgtd, max_len=256, remove_same=True, cache_token=500000000):

	def save_cache(cache, tgtfl):

		ens = "\n".encode("utf-8")

		with FileList(tgtfl, "wb") as wfl:
			for tmp in iter_dict_sort(cache, free=True):
				lines = zip(*tmp)
				for du, f in zip(lines, wfl):
					f.write(ens.join(du))
					f.write(ens)

	_max_len = max(1, max_len - 2)

	_insert_func = dict_insert_set if remove_same else dict_insert_list
	data = {}
	mem_token = curf = 0
	num_files = len(srcfl)
	with FileList(srcfl, "rb") as fl:
		for lines in zip(*fl):
			lines = [line.strip() for line in lines]
			if all(lines):
				lines, lens = zip(*[clean_liststr_lentok(line.decode("utf-8").split()) for line in lines])
				if all_le(lens, max_len):
					lgth = sum(lens)
					data = _insert_func(data, tuple(line.encode("utf-8") for line in lines), lgth, *reversed(lens[1:]))
					mem_token += lgth
					if mem_token >= cache_token:
						save_cache(data, [pjoin(tgtd, "%d.%d.txt" % (i, curf,)) for i in range(num_files)])
						data = {}
						mem_token = 0
						curf += 1
	if data:
		save_cache(data, [pjoin(tgtd, "%d.%d.txt" % (i, curf,)) for i in range(num_files)])

if __name__ == "__main__":
	handle(sys.argv[1:-2], sys.argv[-2], int(sys.argv[-1]))
