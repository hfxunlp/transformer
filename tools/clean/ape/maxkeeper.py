#encoding: utf-8

import sys

from utils.fmt.base import clean_liststr_lentok, sys_open

def handle(srcfs, srcfm, srcft, tgtfs, tgtfm, tgtft, max_len=256):

	_max_len = max(1, max_len - 2)

	data = {}

	with sys_open(srcfs, "rb") as fs, sys_open(srcfm, "rb") as fm, sys_open(srcft, "rb") as ft:
		for ls, lm, lt in zip(fs, fm, ft):
			ls, lm, lt = ls.strip(), lm.strip(), lt.strip()
			if ls and lt:
				ls, slen = clean_liststr_lentok(ls.decode("utf-8").split())
				lm, mlen = clean_liststr_lentok(lm.decode("utf-8").split())
				lt, tlen = clean_liststr_lentok(lt.decode("utf-8").split())
				ls, lm, lt = ls.encode("utf-8"), lm.encode("utf-8"), lt.encode("utf-8")
				if (slen <= _max_len) and (mlen <= _max_len) and (tlen <= _max_len):
					if ls in data:
						data[ls][(lm, lt,)] = data[ls].get((lm, lt,), 0) + 1
					else:
						data[ls] = {(lm, lt,): 1}

	_clean = {}
	for ls, v in data.items():
		if len(v) > 1:
			rlt = []
			_maxf = 0
			for key, value in v.items():
				if value > _maxf:
					_maxf = value
					rlt = [key]
				elif value == _maxf:
					rlt.append(key)
			for lt in rlt:
				if lt in _clean:
					_clean[lt][ls] = _clean[lt].get(ls, 0) + 1
				else:
					_clean[lt] = {ls: 1}
		else:
			lt = list(v.keys())[0]
			if lt in _clean:
				_clean[lt][ls] = _clean[lt].get(ls, 0) + 1
			else:
				_clean[lt] = {ls: 1}

	data = _clean
	ens = "\n".encode("utf-8")
	with sys_open(tgtfs, "wb") as fs, sys_open(tgtfm, "wb") as fm, sys_open(tgtft, "wb") as ft:
		for (lm, lt,), v in data.items():
			if len(v) > 1:
				rls = []
				_maxf = 0
				for key, value in v.items():
					if value > _maxf:
						_maxf = value
						rls = [key]
					elif value == _maxf:
						rls.append(key)
				rlm = ens.join([lm for i in range(len(rls))])
				rlt = ens.join([lt for i in range(len(rls))])
				rls = ens.join(rls)
			else:
				rlm = lm
				rlt = lt
				rls = list(v.keys())[0]
			fs.write(rls)
			fs.write(ens)
			fm.write(rlm)
			fm.write(ens)
			ft.write(rlt)
			ft.write(ens)

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], int(sys.argv[7]))
