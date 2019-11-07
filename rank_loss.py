#encoding: utf-8

# usage: python rank.py rsf h5f models...

norm_token = True

import sys

import torch

from tqdm import tqdm

import h5py

import cnfg.base as cnfg

from transformer.NMT import NMT
from transformer.EnsembleNMT import NMT as Ensemble
from parallel.parallelMT import DataParallelMT
from parallel.base import DataParallelCriterion

from loss import LabelSmoothingLoss

from utils.base import *
from utils.fmt.base4torch import parse_cuda

def load_fixing(module):

	if "fix_load" in dir(module):
		module.fix_load()

td = h5py.File(sys.argv[2], "r")

ntest = td["ndata"][:].item()
nword = td["nword"][:].tolist()
nwordi, nwordt = nword[0], nword[-1]

cuda_device = torch.device(cnfg.gpuid)

if len(sys.argv) == 4:
	mymodel = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.share_emb, cnfg.nhead, cnfg.cache_len, cnfg.attn_hsize, cnfg.norm_output, cnfg.bindDecoderEmb, cnfg.forbidden_indexes)

	mymodel = load_model_cpu(sys.argv[3], mymodel)
	mymodel.apply(load_fixing)

else:
	models = []
	for modelf in sys.argv[3:]:
		tmp = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.share_emb, cnfg.nhead, cnfg.cache_len, cnfg.attn_hsize, cnfg.norm_output, cnfg.bindDecoderEmb, cnfg.forbidden_indexes)

		tmp = load_model_cpu(modelf, tmp)
		tmp.apply(load_fixing)

		models.append(tmp)
	mymodel = Ensemble(models)

mymodel.eval()

lossf = LabelSmoothingLoss(nwordt, cnfg.label_smoothing, ignore_index=0, reduction='none', forbidden_index=cnfg.forbidden_indexes)

use_cuda, cuda_device, cuda_devices, multi_gpu = parse_cuda(cnfg.use_cuda, cnfg.gpuid)

# Important to make cudnn methods deterministic
set_random_seed(cnfg.seed, use_cuda)

if use_cuda:
	mymodel.to(cuda_device)
	lossf.to(cuda_device)
	if multi_gpu:
		mymodel = DataParallelMT(mymodel, device_ids=cuda_devices, output_device=cuda_device.index, host_replicate=True, gather_output=False)
		lossf = DataParallelCriterion(lossf, device_ids=cuda_devices, output_device=cuda_device.index, replicate_once=True)

ens = "\n".encode("utf-8")

src_grp, tgt_grp = td["src"], td["tgt"]
with open(sys.argv[1], "wb") as f:
	with torch.no_grad():
		for i in tqdm(range(ntest)):
			_curid = str(i)
			seq_batch = torch.from_numpy(src_grp[_curid][:]).long()
			seq_o = torch.from_numpy(tgt_grp[_curid][:]).long()
			if use_cuda:
				seq_batch = seq_batch.to(cuda_device)
				seq_o = seq_o.to(cuda_device)
			lo = seq_o.size(1) - 1
			ot = seq_o.narrow(1, 1, lo).contiguous()
			output = mymodel(seq_batch, seq_o.narrow(1, 0, lo))
			loss = lossf(output, ot).sum(-1).view(-1, lo).sum(-1)
			if norm_token:
				lenv = ot.ne(0).int().sum(-1).to(loss)
				loss = loss / lenv
			f.write("\n".join([str(rsu) for rsu in loss.tolist()]).encode("utf-8"))
			loss = output = ot = seq_batch = seq_o = None
			f.write(ens)

td.close()
