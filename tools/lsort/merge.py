#encoding: utf-8

import sys

from utils.fmt.base import clean_liststr_lentok, all_true, all_le, maxfreq_filter, shuffle_pair, iter_dict_sort, dict_insert_list, dict_insert_set, FileList

from random import seed as rpyseed
from os import walk, path

# remove_same: reduce same data in the corpus
# shuf: shuffle the data of same source/target length
# max_remove: if one source has several targets, only keep those with highest frequency

def handle(cached, tgtfl, remove_same=False, shuf=True, max_remove=True):

	def paral_reader(fl):

		with FileList(srcfl, "rb") as fl:
			for lines in zip(*fl):
				lines = [line.strip() for line in lines]
				if all_true(lines):
					lines, lens = zip(*[clean_liststr_lentok(line.decode("utf-8").split()) for line in lines])
					lgth = sum(lens)
					yield tuple(line.encode("utf-8") for line in lines), lgth, *reversed(lens[1:])

	def open_files(cache_dir, num_files):

		rs = []
		query = []
		opened = set()
		for root, dirs, files in walk(cache_dir):
			for file in files:
				curfid = file.split(".")[1]
				if curfid not in opened:
					pg = paral_reader([path.join(cache_dir, "%d.%s.txt" % (i, curfid,)) for i in range(num_files)])
					opened.add(curfid)
					try:
						prd = next(pg)
					except StopIteration:
						prd = None
					if prd:
						rs.append(pg)
						query.append((prd[0], prd[1:],))

		return rs, query

	def update_query(fl, query):

		min_len = None
		rs = None
		rid = 0
		for ind, (du, lens,) in enumerate(query):
			if (min_len is None) or (lens < min_len):
				min_len = lens
				rs = (du, lens,)
				rid = ind
		try:
			_next_v = next(fl[rid])
			query[rid] = (_next_v[0], _next_v[1:],)
		except StopIteration:
			del query[rid]
			del fl[rid]

		return rs, query, fl

	def write_data(data, wfl, ens, shuf=True, max_remove=False):

		lines = zip(*data)
		if len(data) > 1:
			if max_remove:
				lines = maxfreq_filter(*lines)
			if shuf:
				lines = shuffle_pair(*lines)
		for du, f in zip(lines, wfl):
			f.write(ens.join(du))
			f.write(ens)

	num_files = len(tgtfl)
	fl, query = open_files(cached, num_files)

	_dedup_data = remove_same and (not max_remove)
	data = set() if _dedup_data else []
	cur_len = None

	ens = "\n".encode("utf-8")

	with FileList(tgtfl, "wb") as wfl:
		while query:
			rs, query, fl = update_query(fl, query)
			if rs:
				du, lens = rs
				if cur_len is None:
					cur_len = lens
				if lens == cur_len:
					if _dedup_data:
						if not du in data:
							data.add(du)
					else:
						data.append(du)
				else:
					if data:
						write_data(data, wfl, ens, shuf=shuf, max_remove=max_remove)
					cur_len = lens
					data = set([du]) if _dedup_data else [du]
		if data:
			write_data(data, wfl, ens, shuf=shuf, max_remove=max_remove)

if __name__ == "__main__":
	rpyseed(666666)
	handle(sys.argv[1], sys.argv[2:])
