#encoding: utf-8

import sys
from random import seed as rpyseed

from utils.fmt.base import clean_liststr_lentok, maxfreq_filter, shuffle_pair, iter_dict_sort, dict_insert_list

# remove_same: reduce same data in the corpus
# shuf: shuffle the data of same source/target length
# max_remove: if one source has several targets, only keep those with highest frequency

def handle(srcfs, srcft, tgtfs, tgtft, max_len=256, remove_same=False, shuf=True, max_remove=False):

	_max_len = max(1, max_len - 2)

	data = {}

	with open(srcfs, "rb") as fs, open(srcft, "rb") as ft:
		for ls, lt in zip(fs, ft):
			ls, lt = ls.strip(), lt.strip()
			if ls and lt:
				ls, slen = clean_liststr_lentok(ls.decode("utf-8").split())
				lt, tlen = clean_liststr_lentok(lt.decode("utf-8").split())
				if (slen <= _max_len) and (tlen <= _max_len):
					lgth = slen + tlen
					data = dict_insert_list(data, (ls, lt,), lgth, tlen)

	ens = "\n".encode("utf-8")

	with open(tgtfs, "wb") as fs, open(tgtft, "wb") as ft:
		for tmp in iter_dict_sort(data):
			ls, lt = zip(*tmp)
			if len(ls) > 1:
				if remove_same:
					ls, lt = maxfreq_filter(ls, lt, max_remove)
				if shuf:
					ls, lt = shuffle_pair(ls, lt)
			fs.write("\n".join(ls).encode("utf-8"))
			fs.write(ens)
			ft.write("\n".join(lt).encode("utf-8"))
			ft.write(ens)

if __name__ == "__main__":
	rpyseed(666666)
	handle(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
