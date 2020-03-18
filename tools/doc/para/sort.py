#encoding: utf-8

import sys
from random import seed as rpyseed

from utils.fmt.base import clean_liststr_lentok, maxfreq_filter, shuffle_pair, iter_dict_sort, dict_insert_set

def handle(srcfs, srcft, tgtfs, tgtft, remove_same=False, shuf=True, max_remove=False):

	data = {}
	cache = []
	mxtoks = mxtokt = ntoks = ntokt = 0

	with open(srcfs, "rb") as fs, open(srcft, "rb") as ft:
		for ls, lt in zip(fs, ft):
			ls, lt = ls.strip(), lt.strip()
			if ls and lt:
				ls, slen = clean_liststr_lentok(ls.decode("utf-8").split())
				lt, tlen = clean_liststr_lentok(lt.decode("utf-8").split())
				cache.append((ls, lt,))
				if slen > mxtoks:
					mxtoks = slen
				if tlen > mxtokt:
					mxtokt = tlen
				ntoks += slen
				ntokt += tlen
			else:
				if cache:
					nsent = len(cache)
					ls, lt = zip(*cache)
					_tmp = ("\n".join(ls), "\n".join(lt),)
					data = dict_insert_set(data, _tmp, nsent, mxtoks + mxtokt, mxtokt, ntoks + ntokt, ntokt)
					cache = []
					mxtoks = mxtokt = ntoks = ntokt = 0
		if cache:
			nsent = len(cache)
			ls, lt = zip(*cache)
			_tmp = ("\n".join(ls), "\n".join(lt),)
			data = dict_insert_set(data, _tmp, nsent, mxtoks + mxtokt, mxtokt, ntoks + ntokt, ntokt)
			cache = []
			mxtoks = mxtokt = ntoks = ntokt = 0

	ens = "\n\n".encode("utf-8")

	with open(tgtfs, "wb") as fs, open(tgtft, "wb") as ft:
		for tmp in iter_dict_sort(data):
			ls, lt = zip(*tmp)
			if len(ls) > 1:
				if remove_same:
					ls, lt = maxfreq_filter(ls, lt, max_remove)
				if shuf:
					ls, lt = shuffle_pair(ls, lt)
			fs.write("\n\n".join(ls).encode("utf-8"))
			fs.write(ens)
			ft.write("\n\n".join(lt).encode("utf-8"))
			ft.write(ens)

if __name__ == "__main__":
	rpyseed(666666)
	handle(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
