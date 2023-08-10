#encoding: utf-8

import sys

from utils.fmt.base import FileList, all_gt, clean_list

def handle(srcfl, tgtfl, r=0.4):

	ens = "\n".encode("utf-8")
	with FileList(srcfl, "rb") as rfl, FileList(tgtfl, "wb") as wfl:
		for lines in zip(*rfl):
			lines = [line.strip() for line in lines]
			if all(lines):
				lines = [clean_list(line.decode("utf-8").split()) for line in lines]
				ratios = [float(len(set(line))) / float(len(line)) for line in lines]
				if all_gt(ratios, r):
					for line, f in zip(lines, wfl):
						f.write(" ".join(line).encode("utf-8"))
						f.write(ens)

if __name__ == "__main__":

	_nargs = len(sys.argv)
	if _nargs % 2 == 0:
		_sep_ind = _nargs // 2
		handle(sys.argv[1:_sep_ind], sys.argv[_sep_ind:-1], r=float(sys.argv[-1]))
	else:
		_sep_ind = (_nargs + 1) // 2
		handle(sys.argv[1:_sep_ind], sys.argv[_sep_ind:])
