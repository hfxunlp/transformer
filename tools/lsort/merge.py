#encoding: utf-8

import sys

from utils.fmt.base import clean_liststr_lentok, maxfreq_filter, shuffle_pair, iter_dict_sort

from random import seed as rpyseed
from os import walk, path

# remove_same: reduce same data in the corpus
# shuf: shuffle the data of same source/target length
# max_remove: if one source has several targets, only keep those with highest frequency

def handle(cached, srcf, rsf, remove_same=True, shuf=True, max_remove=True):

	def paral_reader(fsrc, ftgt):
		srcf, tgtf = open(fsrc, "rb"), open(ftgt, "rb")
		src, tgt = srcf.readline(), tgtf.readline()
		while src and tgt:
			src, tgt = src.strip(), tgt.strip()
			if src and tgt:
				src, lsrc = clean_liststr_lentok(src.decode("utf-8").split())
				tgt, ltgt = clean_liststr_lentok(tgt.decode("utf-8").split())
				yield src, tgt, ltgt + lsrc, ltgt
			src, tgt = srcf.readline(), tgtf.readline()
		srcf.close()
		tgtf.close()

	def open_files(cache_dir):
		rs = []
		query = []
		for root, dirs, files in walk(cache_dir):
			for file in files:
				if file.endswith(".src"):
					srcf = path.join(root, file)
					tgtf = path.join(root, file.replace(".src", ".tgt"))
					pg = paral_reader(srcf, tgtf)
					try:
						prd = next(pg)
					except StopIteration:
						prd = None
					if prd:
						rs.append(pg)
						query.append(prd)
		return rs, query

	def update_query(fl, query, max_len):

		min_len = max_len
		rs = None
		rid = 0
		for ind, (src, tgt, l, lt) in enumerate(query):
			if l < min_len:
				min_len = l
				rs = (src, tgt, l, lt)
				rid = ind
		try:
			query[rid] = next(fl[rid])
		except StopIteration:
			del query[rid]
			del fl[rid]

		return rs, query, fl

	def write_data(data, fs, ft, ens, rsame, shuf, mclean):

		for tmp in iter_dict_sort(data):
			ls, lt = zip(*tmp)
			if len(ls) > 1:
				if rsame:
					ls, lt = maxfreq_filter(ls, lt, mclean)
				if shuf:
					ls, lt = shuffle_pair(ls, lt)
			fs.write("\n".join(ls).encode("utf-8"))
			fs.write(ens)
			ft.write("\n".join(lt).encode("utf-8"))
			ft.write(ens)

	fl, query = open_files(cached)
	maxlen = float("inf")

	data = {}
	cur_len = 0

	ens = "\n".encode("utf-8")

	with open(srcf, "wb") as fs, open(rsf, "wb") as ft:
		while query:
			rs, query, fl = update_query(fl, query, maxlen)
			if rs:
				src, tgt, l, lt = rs
				if l == cur_len:
					if lt in data:
						data[lt].append((src, tgt,))
					else:
						data[lt] = [(src, tgt,)]
				else:
					if data:
						write_data(data, fs, ft, ens, remove_same, shuf, max_remove)
					cur_len = l
					data = {lt: [(src, tgt,)]}
			else:
				break
		if data:
			write_data(data, fs, ft, ens, remove_same, shuf, max_remove)

if __name__ == "__main__":
	rpyseed(666666)
	handle(sys.argv[1], sys.argv[2], sys.argv[3])
