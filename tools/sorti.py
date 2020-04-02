#encoding: utf-8

import sys

from utils.fmt.base import clean_liststr_lentok, iter_dict_sort

def handle(srcfs, tgtfs, max_len=1048576):

	data = {}

	_max_len = max(1, max_len - 2)

	with open(srcfs, "rb") as fs:
		for ls in fs:
			ls = ls.strip()
			if ls:
				ls, lgth = clean_liststr_lentok(ls.decode("utf-8").split())
				if lgth <= _max_len:
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
	handle(sys.argv[1], sys.argv[2]) if len(sys.argv) == 3 else handle(sys.argv[1], sys.argv[2], int(sys.argv[-1]))
