#encoding: utf-8

import sys
from random import seed as rpyseed, shuffle

from utils.fmt.base import FileList, clean_str, sys_open

def handle(srcfl, rsfl):

	data = []
	with FileList(srcfl, "rb") as files:
		for lines in zip(*files):
			data.append([clean_str(tmpu.strip().decode("utf-8")) for tmpu in lines])

	shuffle(data)

	ens = "\n".encode("utf-8")
	for du, rsf in zip(zip(*data), rsfl):
		with sys_open(rsf, "wb") as fwrt:
			fwrt.write("\n".join(du).encode("utf-8"))
			fwrt.write(ens)

if __name__ == "__main__":
	rpyseed(666666)
	_ind = (len(sys.argv) + 1) // 2
	handle(sys.argv[1:_ind], sys.argv[_ind:])
