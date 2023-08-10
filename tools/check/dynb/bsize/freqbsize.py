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

def handle(srcf, minv, delt):

	rsd = {}
	for data in load_log(srcf):
		tmp = floor((int(data[-1]) - minv) / delt)
		rsd[tmp] = rsd.get(tmp, 0) + 1

	print(rsd)

if __name__ == "__main__":
	handle(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
