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

def handle(srcf):

	rsl = []
	for data in load_log(srcf):
		tmp = int(data[-1])
		rsl.append(tmp)
	axtmp, intmp = [], []
	for gd in giter(rsl):
		print("%.2f" % (sum(gd) / float(len(gd)),))
		axtmp.append(max(gd))
		#print(max(gd))
		intmp.append(min(gd))
		#print(min(gd))

	print(max(axtmp), min(intmp))

if __name__ == "__main__":
	handle(sys.argv[1])
