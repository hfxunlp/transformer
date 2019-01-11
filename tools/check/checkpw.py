#encoding: utf-8

'''
	example usage:
	ln tools/check/checkpw.py .
	python checkpw.py expm/debug/checkpoint.t7 un-cache/tgt.vcb 
'''

import sys

import torch

import h5py

import cnfg

from transformer.NMT import NMT

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

td = h5py.File(cnfg.test_data, "r")
nwordi = int(td["nwordi"][:][0])
td.close()

vcbt, nwordt = ldvocab(sys.argv[2])
vcbt = reverse_dict(vcbt)

mymodel = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.share_emb, cnfg.nhead, cnfg.cache_len, cnfg.attn_hsize)

mymodel.load_state_dict(torch.load(sys.argv[1], map_location='cpu'))

initmodel = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.share_emb, cnfg.nhead, cnfg.cache_len, cnfg.attn_hsize)

print(initmodel.enc.pemb.w.data.equal(initmodel.dec.pemb.w.data))
print(initmodel.enc.pemb.w.data.equal(mymodel.enc.pemb.w.data))
print(initmodel.enc.pemb.w.data.equal(mymodel.dec.pemb.w.data))
