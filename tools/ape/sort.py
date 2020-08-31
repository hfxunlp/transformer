#encoding: utf-8

import sys
from random import seed as rpyseed

from utils.fmt.base import clean_liststr_lentok, maxfreq_filter, shuffle_pair, iter_dict_sort, dict_insert_list

# remove_same: reduce same data in the corpus
# shuf: shuffle the data of same source/target length
# max_remove: if one source has several targets, only keep those with highest frequency

def handle(srcfs, srcfm, srcft, tgtfs, tgtfm, tgtft, max_len=256, remove_same=False, shuf=True, max_remove=False):

	_max_len = max(1, max_len - 2)

	data = {}

	with open(srcfs, "rb") as fs, open(srcfm, "rb") as fm, open(srcft, "rb") as ft:
		for ls, lm, lt in zip(fs, fm, ft):
			ls, lm, lt = ls.strip(), lm.strip(), lt.strip()
			if ls and lm and lt:
				ls, slen = clean_liststr_lentok(ls.decode("utf-8").split())
				lm, mlen = clean_liststr_lentok(lm.decode("utf-8").split())
				lt, tlen = clean_liststr_lentok(lt.decode("utf-8").split())
				if (slen <= _max_len) and (mlen <= _max_len) and (tlen <= _max_len):
					data = dict_insert_list(data, (ls, lm, lt,), slen + tlen + mlen, tlen, mlen)

	ens = "\n".encode("utf-8")

	with open(tgtfs, "wb") as fs, open(tgtfm, "wb") as fm, open(tgtft, "wb") as ft:
		for tmp in iter_dict_sort(data):
			ls, lm, lt = zip(*tmp)
			if len(ls) > 1:
				if remove_same:
					(ls, lm,), lt = maxfreq_filter((ls, lm,), lt, max_remove)
				if shuf:
					ls, lm, lt = shuffle_pair(ls, lm, lt)
			fs.write("\n".join(ls).encode("utf-8"))
			fs.write(ens)
			fm.write("\n".join(lm).encode("utf-8"))
			fm.write(ens)
			ft.write("\n".join(lt).encode("utf-8"))
			ft.write(ens)

if __name__ == "__main__":
	rpyseed(666666)
	handle(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], int(sys.argv[7]))
