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

def handle(srcf):

	rsl = []
	for data in load_log(srcf):
		tmp = int(data[-1])
		rsl.append(tmp)

	print(min(rsl), "%.2f" % (sum(rsl) / float(len(rsl)),), max(rsl))

if __name__ == "__main__":
	handle(sys.argv[1])
