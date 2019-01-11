#encoding: utf-8

import sys

import torch

from tqdm import tqdm

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

def eva(ed, i, model):
	bid = str(i)
	seq_batch = torch.from_numpy(ed["i"+bid][:]).long()
	seq_o = torch.from_numpy(ed["t"+bid][:]).long()
	lo = seq_o.size(1) - 1
	if use_cuda:
		seq_batch = seq_batch.to(cuda_device)
		seq_o = seq_o.to(cuda_device)
	output = model(seq_batch, seq_o.narrow(1, 0, lo))
	_, trans = torch.max(output, -1)
	return trans

td = h5py.File(cnfg.test_data, "r")

ntest = int(td["ndata"][:][0])
nwordi = int(td["nwordi"][:][0])
vcbt, nwordt = ldvocab(sys.argv[2])
vcbt = reverse_dict(vcbt)

cuda_device = torch.device(cnfg.gpuid)

mymodel = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.share_emb, cnfg.nhead, cnfg.cache_len, cnfg.attn_hsize)

mymodel.load_state_dict(torch.load(sys.argv[1], map_location='cpu'))

mymodel.eval()

use_cuda = cnfg.use_cuda
if use_cuda and torch.cuda.is_available():
	use_cuda = True

if use_cuda:
	mymodel.to(cuda_device)

beam_size = cnfg.beam_size

ens = "\n".encode("utf-8")

with open(sys.argv[3], "wb") as f:
	for i in tqdm(range(ntest)):
		seq_batch = torch.from_numpy(td["i" + str(i)][:]).long()
		if use_cuda:
			seq_batch = seq_batch.to(cuda_device)
		#output = mymodel.decode(seq_batch, beam_size).tolist()
		output = eva(td, i, mymodel).tolist()
		for tran in output:
			tmp = []
			for tmpu in tran:
				if (tmpu == 2) or (tmpu == 0):
					break
				else:
					tmp.append(vcbt[tmpu])
			f.write(" ".join(tmp).encode("utf-8"))
			f.write(ens)

td.close()
