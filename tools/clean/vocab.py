#encoding: utf-8

import sys

def list_reader(fname):
	def clear_list(lin):
		rs = []
		for tmpu in lin:
			if tmpu:
				rs.append(tmpu)
		return rs
	with open(fname, "rb") as frd:
		for line in frd:
			tmp = line.strip()
			if tmp:
				tmp = clear_list(tmp.decode("utf-8").split())
				yield tmp

def ldvocab(vfile, minf = False, omit_vsize = False):
	rs = []
	if omit_vsize:
		vsize = omit_vsize
	else:
		vsize = False
	cwd = 0
	for data in list_reader(vfile):
		freq = int(data[0])
		if (not minf) or freq > minf:
			if vsize:
				ndata = len(data) - 1
				if vsize >= ndata:
					rs.extend(data[1:])
					cwd += ndata
				else:
					rs.extend(data[1:vsize + 1])
					cwd += vsize
					break
				vsize -= ndata
				if vsize <= 0:
					break
			else:
				rs.extend(data[1:])
				cwd += len(data) - 1
		else:
			break
	return rs, cwd

# vratio: percentages of vocabulary size of retrieved words of least frequencies
# dratio: a datum will be dropped who contains high frequency words less than this ratio

def handle(srcfs, srcft, tgtfs, tgtft, vcbfs, vcbft, vratio, dratio=None):

	def legal(sent, ilgset, ratio):

		total = 0
		ilg = 0
		for tmpu in sent.split():
			if tmpu:
				if tmpu in ilgset:
					ilg += 1
				total += 1
		rt = float(ilg) / float(total)

		return False if rt > ratio else True

	_dratio = vratio if dratio is None else dratio

	ens = "\n".encode("utf-8")

	vcbs, nvs = ldvocab(vcbfs)
	vcbt, nvt = ldvocab(vcbft)
	ilgs = set(vcbs[int(float(nvs) * (1.0 - vratio)):])
	ilgt = set(vcbt[int(float(nvt) * (1.0 - vratio)):])

	with open(srcfs, "rb") as fs, open(srcft, "rb") as ft, open(tgtfs, "wb") as fsw, open(tgtft, "wb") as ftw:
		total = 0
		keep = 0
		for ls, lt in zip(fs, ft):
			ls, lt = ls.strip(), lt.strip()
			if ls and lt:
				ls, lt = ls.decode("utf-8"), lt.decode("utf-8")
				if legal(ls, ilgs, _dratio) and legal(lt, ilgt, _dratio):
					fsw.write(ls.encode("utf-8"))
					fsw.write(ens)
					ftw.write(lt.encode("utf-8"))
					ftw.write(ens)
					keep += 1
				total += 1
		print("%d in %d data keeped with ratio %.2f" % (keep, total, float(keep) / float(total) * 100.0 if total > 0 else 0.0))

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], float(sys.argv[7]))
