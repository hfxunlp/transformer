#encoding: utf-8

import sys

from math import floor

def load_log(fname):

	def legal(clin):

		if clin[-1][0] != "D" and clin[0][0].isalpha() and clin[0][1].isdigit() and len(clin[-1].split()) == 1:
			return True
		else:
			return False

	cache = []
	with open(fname, "rb") as frd:
		for line in frd:
			tmp = line.strip()
			if tmp:
				tmp = tmp.decode("utf-8")
				if tmp == "ES":
					if cache:
						if legal(cache):
							yield cache
					cache = []
				elif tmp[0].isalpha() and tmp[1].isdigit():
					if cache:
						if legal(cache):
							yield cache
						cache = [tmp]
				else:
					cache.append(tmp)
		if cache:
			if legal(cache):
				yield cache

def giter(lin, nsep=20):

	_nd = len(lin)
	_nuni = max(floor(_nd / nsep), 1)
	_dnuni = _nuni * 2
	lind = 0
	while lind < _nd:
		if lind + _dnuni > _nd:
			rind = _nd
		else:
			rind = lind + _nuni
		yield lin[lind:rind]
		lind = rind

def retrieve_con(lin):

	return [float(tmpu.split()[-1]) for tmpu in lin[1:-1]]

def handle(srcf):

	rsl = []
	for data in load_log(srcf):
		tmp = retrieve_con(data)
		#rsl.append(max(tmp))
		rsl.append(min(tmp))
		#rsl.append(sum(tmp) / len(tmp))
	for gd in giter(rsl):
		print("%.2f" % (sum(gd) / float(len(gd)),))

if __name__ == "__main__":
	handle(sys.argv[1])
