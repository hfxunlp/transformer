#encoding: utf-8

import sys
from random import shuffle
from random import seed as rpyseed

# remove_same: reduce same data in the corpus
# shuf: shuffle the data of same source/target length
# max_remove: if one source has several targets, only keep those with highest frequency

def handle(srcfs, srcft, tgtfs, tgtft, max_len=256, remove_same=False, shuf=True, max_remove=False):

	def clean(lin):
		rs = []
		for lu in lin:
			if lu:
				rs.append(lu)
		return " ".join(rs), len(rs)

	def filter(ls, lt, max_remove=True):
		tmp = {}
		for us, ut in zip(ls, lt):
			if us not in tmp:
				tmp[us] = {ut: 1} if max_remove else set([ut])
			else:
				if max_remove:
					tmp[us][ut] = tmp[us].get(ut, 0) + 1
				elif ut not in tmp[us]:
					tmp[us].add(ut)
		rls, rlt = [], []
		if max_remove:
			for tus, tlt in tmp.items():
				_rs = []
				_maxf = 0
				for key, value in tlt.items():
					if value > _maxf:
						_maxf = value
						_rs = [key]
					elif value == _maxf:
						_rs.append(key)
				for tut in _rs:
					rls.append(tus)
					rlt.append(tut)
		else:
			for tus, tlt in tmp.items():
				for tut in tlt:
					rls.append(tus)
					rlt.append(tut)
		return rls, rlt

	def shuffle_pair(ls, lt):
		tmp = list(zip(ls, lt))
		shuffle(tmp)
		rs, rt = zip(*tmp)
		return rs, rt

	_max_len = max(1, max_len - 2)

	data = {}

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

	length = list(data.keys())
	length.sort()

	ens = "\n".encode("utf-8")

	with open(tgtfs, "wb") as fs, open(tgtft, "wb") as ft:
		for lgth in length:
			lg = list(data[lgth].keys())
			lg.sort()
			for lu in lg:
				ls, lt = zip(*data[lgth][lu])
				if len(ls) > 1:
					if remove_same:
						ls, lt = filter(ls, lt, max_remove)
					if shuf:
						ls, lt = shuffle_pair(ls, lt)
				fs.write("\n".join(ls).encode("utf-8"))
				fs.write(ens)
				ft.write("\n".join(lt).encode("utf-8"))
				ft.write(ens)

if __name__ == "__main__":
	rpyseed(666666)
	handle(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
