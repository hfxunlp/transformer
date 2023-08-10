#encoding: utf-8

import sys
from math import floor

from utils.fmt.base import sys_open

def load_log(fname):

	def legal(clin):

		if clin[-1][0] != "D" and clin[0][0].isalpha() and clin[0][1].isdigit() and len(clin[-1].split()) == 1:
			return True
		else:
			return False

	cache = []
	with sys_open(fname, "rb") as frd:
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

def count(lin):

	rsd = {}
	for lu in lin:
		rsd[lu] = rsd.get(lu, 0) + 1
	return rsd

def normd(din, total):

	rsd = {}
	_t = float(total)
	for k, v in din.items():
		rsd[k] = v / _t

	return rsd

def handle(srcf):

	rsl = []
	for data in load_log(srcf):
		tmp = data[0].split()[0]
		rsl.append(tmp)
	for gd in giter(rsl):
		print(normd(count(gd), len(gd)))

if __name__ == "__main__":
	handle(sys.argv[1])
