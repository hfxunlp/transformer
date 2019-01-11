#encoding: utf-8

import sys

def handle(srcfs, srcft, tgtfs, tgtft, max_len=256):

	def clean(lin):
		rs = []
		for lu in lin:
			if lu:
				rs.append(lu)
		return " ".join(rs), len(rs)

	_max_len = max(1, max_len - 2)

	data = {}

	with open(srcfs, "rb") as fs, open(srcft, "rb") as ft:
		for ls, lt in zip(fs, ft):
			ls, lt = ls.strip(), lt.strip()
			if ls and lt:
				ls, slen = clean(ls.decode("utf-8").split())
				lt, tlen = clean(lt.decode("utf-8").split())
				if (slen <= _max_len) and (tlen <= _max_len):
					if ls in data:
						data[ls][lt] = data[ls].get(lt, 0) + 1
					else:
						data[ls] = {lt: 1}

	ens = "\n".encode("utf-8")

	with open(tgtfs, "wb") as fs, open(tgtft, "wb") as ft:
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
				rls = "\n".join([ls for i in range(len(rlt))])
				rlt = "\n".join(rlt)
			else:
				rls = ls
				rlt = list(v.keys())[0]
			fs.write(rls.encode("utf-8"))
			fs.write(ens)
			ft.write(rlt.encode("utf-8"))
			ft.write(ens)

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
