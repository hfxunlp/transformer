#encoding: utf-8

import sys

from utils.fmt.base import clean_list_len, iter_dict_sort

def handle(srcfs, tgtfs):

	data = {}

	with open(srcfs, "rb") as fs:
		for ls in fs:
			ls = ls.strip()
			if ls:
				ls, lgth = clean_list_len(ls.decode("utf-8").split())
				if lgth in data:
					if ls not in data[lgth]:
						data[lgth].add(ls)
				else:
					data[lgth] = set([ls])

	ens = "\n".encode("utf-8")

	with open(tgtfs, "wb") as fs:
		for tmp in iter_dict_sort(data):
			fs.write("\n".join(tmp).encode("utf-8"))
			fs.write(ens)

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2])
