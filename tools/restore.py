#encoding: utf-8

import sys

# WARNING: all() might be too strict in some cases which may use any()

from utils.fmt.base import FileList, clean_str, sys_open

# srtfl: (k - 1) source + 1 target
def handle(srcfl, srtfl, tgtf):

	data = {}

	with FileList(srtfl, "rb") as fs:
		for lines in zip(*fs):
			lines = tuple(line.strip() for line in lines)
			if all(lines):
				lines = tuple(clean_str(line.decode("utf-8")) for line in lines)
				data[lines[:-1]] = lines[-1].encode("utf-8")

	ens = "\n".encode("utf-8")
	with FileList(srcfl, "rb") as fs, sys_open(tgtf, "wb") as ft:
		for lines in zip(*fs):
			lines = tuple(line.strip() for line in lines)
			if all(lines):
				lines = tuple(clean_str(line.decode("utf-8")) for line in lines)
				if lines in data:
					ft.write(data[lines])
			ft.write(ens)

if __name__ == "__main__":
	_ind = (len(sys.argv) - 1) // 2
	handle(sys.argv[1:_ind], sys.argv[_ind:-1], sys.argv[-1])
