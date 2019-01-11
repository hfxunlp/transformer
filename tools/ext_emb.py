#encoding: utf-8

''' usage:
	python tools/ext_emb.py vocab_file emb_file result
'''

import sys

import torch

has_unk = True

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

def ldemb(vcb, embf):

	rs = {}
	with open(embf, "rb") as frd:
		for line in frd:
			tmp = line.strip()
			if tmp:
				tmp = tmp.decode("utf-8").split()
				wd = tmp[0]
				if wd in vcb or wd == "<unk>":
					rs[wd] = torch.tensor([float(_t) for _t in tmp[1:]])

	return rs

def reverse_dict(din):
	rs = {}
	for k, v in din.items():
		rs[v] = k
	return rs

def handle(vcbf, embf, rsf):

	vcb, nwd = ldvocab(vcbf)
	emb = ldemb(vcb, embf)
	unkemb = emb.get("<unk>", torch.zeros(emb[list(emb.keys())[0]].size(0)))
	vcb = reverse_dict(vcb)
	rs = []
	for i in range(nwd):
		rs.append(emb.get(vcb[i], unkemb))
	torch.save(torch.stack(rs, 0), rsf)

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3])
