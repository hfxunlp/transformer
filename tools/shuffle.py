#encoding: utf-8

import sys

from random import seed as rpyseed
from random import shuffle

from utils.fmt.base import clean_str

def handle(srcfl, rsfl):

	files = [open(srcf, "rb") for srcf in srcfl]
	data = []
	for lines in zip(*files):
		data.append([clean_str(tmpu.strip().decode("utf-8")) for tmpu in lines])
	for frd in files:
		frd.close()
	shuffle(data)
	files = [open(rsf, "wb") for rsf in rsfl]
	ens = "\n".encode("utf-8")
	for du, fwrt in zip(zip(*data), files):
		fwrt.write("\n".join(du).encode("utf-8"))
		fwrt.write(ens)
		fwrt.close()

if __name__ == "__main__":
	rpyseed(666666)
	_ind = (len(sys.argv) + 1) // 2
	handle(sys.argv[1:_ind], sys.argv[_ind:])
