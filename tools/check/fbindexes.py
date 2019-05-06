#encoding: utf-8

import sys

has_unk = True

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
	global has_unk
	if has_unk:
		rs = {"<pad>":0, "<sos>":1, "<eos>":2, "<unk>":3}
		cwd = 4
	else:
		rs = {"<pad>":0, "<sos>":1, "<eos>":2}
		cwd = 3
	if omit_vsize:
		vsize = omit_vsize
	else:
		vsize = False
	for data in list_reader(vfile):
		freq = int(data[0])
		if (not minf) or freq > minf:
			if vsize:
				ndata = len(data) - 1
				if vsize >= ndata:
					for wd in data[1:]:
						rs[wd] = cwd
						cwd += 1
				else:
					for wd in data[1:vsize + 1]:
						rs[wd] = cwd
						cwd += 1
						ndata = vsize
					break
				vsize -= ndata
				if vsize <= 0:
					break
			else:
				for wd in data[1:]:
					rs[wd] = cwd
					cwd += 1
		else:
			break
	return rs, cwd

def handle(vcbf, srcfl, rsf, minfreq = False, vsize = False):

	vcb, nwords = ldvocab(vcbf, minfreq, vsize)

	fvcb = set(["<pad>", "<sos>", "<eos>", "<unk>"])

	for srcf in srcfl:
		with open(srcf, "rb") as f:
			for line in f:
				tmp = line.strip()
				if tmp:
					for token in tmp.decode("utf-8").split():
						if token and (token not in fvcb):
							fvcb.add(token)

	rsl = []
	for wd, ind in vcb.items():
		if wd not in fvcb:
			rsl.append(ind)

	with open(rsf, "wb") as f:
		f.write("#encoding: utf-8\n\nfbl = ".encode("utf-8"))
		f.write(repr(rsl).encode("utf-8"))
		f.write("\n".encode("utf-8"))

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2:-1], sys.argv[-1])
