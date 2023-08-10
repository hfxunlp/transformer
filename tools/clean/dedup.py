#encoding: utf-8

import sys

from utils.fmt.base import FileList, all_le, clean_liststr_lentok

def handle(srcfl, tgtfl, max_len=256, drop_tail=False):

	_max_len = max(1, max_len - 2)
	data = set()
	ens = "\n".encode("utf-8")

	with FileList(srcfl, "rb") as frl, FileList(tgtfl, "wb") as fwl:
		if drop_tail:
			for lines in zip(*frl):
				lines = [line.strip() for line in lines]
				if all(lines):
					lines, lens = zip(*[clean_liststr_lentok(line.decode("utf-8").split()) for line in lines])
					if all_le(lens, max_len):
						tmp = lines[0].encode("utf-8")
						if tmp not in data:
							for du, f in zip(lines, fwl):
								f.write(du.encode("utf-8"))
								f.write(ens)
							data.add(tmp)
		else:
			for lines in zip(*frl):
				lines = [line.strip() for line in lines]
				if all(lines):
					lines, lens = zip(*[clean_liststr_lentok(line.decode("utf-8").split()) for line in lines])
					if all_le(lens, max_len):
						lines = tuple(line.encode("utf-8") for line in lines)
						if lines not in data:
							for du, f in zip(lines, fwl):
								f.write(du)
								f.write(ens)
							data.add(lines)

if __name__ == "__main__":
	_nargs = len(sys.argv)
	if _nargs % 2 == 0:
		_sep_ind = _nargs // 2
		handle(sys.argv[1:_sep_ind], sys.argv[_sep_ind:-1], max_len=int(sys.argv[-1]))
	else:
		_sep_ind = (_nargs + 1) // 2
		handle(sys.argv[1:_sep_ind], sys.argv[_sep_ind:])
