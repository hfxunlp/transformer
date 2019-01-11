#encoding: utf-8

import sys

import numpy, h5py
import torch

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
	rs = {"<pad>":0, "<unk>":1, "<eos>":2, "<sos>":3}
	cwd = 4
	vsize = False
	if omit_vsize:
		vsize = omit_vsize
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

def reverse_dict(din):
	rs = {}
	for k, v in din.items():
		rs[v] = k
	return rs

def handle(h5f, vcbsf, vcbtf, rsfs, rsft):

	td = h5py.File(h5f, "r")

	ntest = int(td["ndata"][:][0])
	nwordi = int(td["nwordi"][:][0])
	vcbs, nwords = ldvocab(vcbsf)
	vcbs = reverse_dict(vcbs)
	vcbt, nwordt = ldvocab(vcbtf)
	vcbt = reverse_dict(vcbt)

	ens = "\n".encode("utf-8")

	with open(rsfs, "wb") as fs:
		with open(rsft, "wb") as ft:
			for i in range(ntest):
				curd = torch.from_numpy(td["i" + str(i)][:]).long().tolist()
				md = []
				for iu in curd:
					md.append(" ".join([vcbs.get(i) for i in iu]))
				fs.write("\n".join(md).encode("utf-8"))
				fs.write(ens)
				curd = torch.from_numpy(td["t" + str(i)][:]).long().tolist()
				md = []
				for tu in curd:
					md.append(" ".join([vcbt.get(i) for i in tu]))
				ft.write("\n".join(md).encode("utf-8"))
				ft.write(ens)

	td.close()

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
