#encoding: utf-8

import sys

def handle(srcfs, tgtfs):

	def clean(lin):
		rs = []
		for lu in lin:
			if lu:
				rs.append(lu)
		return " ".join(rs), len(rs)

	data = {}

	with open(srcfs, "rb") as fs:
		for ls in fs:
			ls = ls.strip()
			if ls:
				ls, lgth = clean(ls.decode("utf-8").split())
				if lgth not in data:
					data[lgth] = set([ls])
				else:
					if ls not in data[lgth]:
						data[lgth].add(ls)

	length = list(data.keys())
	length.sort()

	ens = "\n".encode("utf-8")

	with open(tgtfs, "wb") as fs:
		for lgth in length:
			fs.write("\n".join(data[lgth]).encode("utf-8"))
			fs.write(ens)

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2])
