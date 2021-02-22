#encoding: utf-8

# usage: python rank.py rsf h5f models...

norm_token = True

import sys

import torch
from torch.cuda.amp import autocast

from tqdm import tqdm

import h5py

import cnfg.base as cnfg
from cnfg.ihyp import *

from transformer.NMT import NMT
from transformer.EnsembleNMT import NMT as Ensemble
from parallel.parallelMT import DataParallelMT
from parallel.base import DataParallelCriterion

from loss.base import LabelSmoothingLoss

from utils.base import *
from utils.fmt.base import pad_id
from utils.fmt.base4torch import parse_cuda

def load_fixing(module):

	if hasattr(module, "fix_load"):
		module.fix_load()

td = h5py.File(sys.argv[2], "r")

ntest = td["ndata"][:].item()
nword = td["nword"][:].tolist()
nwordi, nwordt = nword[0], nword[-1]

if len(sys.argv) == 4:
	mymodel = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.share_emb, cnfg.nhead, cache_len_default, cnfg.attn_hsize, cnfg.norm_output, cnfg.bindDecoderEmb, cnfg.forbidden_indexes)

	mymodel = load_model_cpu(sys.argv[3], mymodel)
	mymodel.apply(load_fixing)

else:
	models = []
	for modelf in sys.argv[3:]:
		tmp = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.share_emb, cnfg.nhead, cache_len_default, cnfg.attn_hsize, cnfg.norm_output, cnfg.bindDecoderEmb, cnfg.forbidden_indexes)

		tmp = load_model_cpu(modelf, tmp)
		tmp.apply(load_fixing)

		models.append(tmp)
	mymodel = Ensemble(models)

mymodel.eval()

lossf = LabelSmoothingLoss(nwordt, cnfg.label_smoothing, ignore_index=pad_id, reduction='none', forbidden_index=cnfg.forbidden_indexes)

use_cuda, cuda_device, cuda_devices, multi_gpu = parse_cuda(cnfg.use_cuda, cnfg.gpuid)
use_amp = cnfg.use_amp and use_cuda

# Important to make cudnn methods deterministic
set_random_seed(cnfg.seed, use_cuda)

if cuda_device:
	mymodel.to(cuda_device)
	lossf.to(cuda_device)
	if multi_gpu:
		mymodel = DataParallelMT(mymodel, device_ids=cuda_devices, output_device=cuda_device.index, host_replicate=True, gather_output=False)
		lossf = DataParallelCriterion(lossf, device_ids=cuda_devices, output_device=cuda_device.index, replicate_once=True)

ens = "\n".encode("utf-8")

src_grp, tgt_grp = td["src"]["4"], td["tgt"]["4"]
with open(sys.argv[1], "wb") as f:
	with torch.no_grad():
		for i in tqdm(range(ntest)):
			_curid = str(i)
			seq_batch = torch.from_numpy(src_grp[_curid][:])
			seq_o = torch.from_numpy(tgt_grp[_curid][:])
			bsize, nsent = seq_batch.size()[:2]
			ebsize = bsize * nsent
			if cuda_device:
				seq_batch = seq_batch.to(cuda_device)
				seq_o = seq_o.to(cuda_device)
			seq_batch, seq_o = seq_batch.long(), seq_o.long()
			lo = seq_o.size(-1) - 1
			ot = seq_o.narrow(-1, 1, lo).contiguous()
			with autocast(enabled=use_amp):
				output = mymodel(seq_batch.view(ebsize, -1), seq_o.narrow(-1, 0, lo).contiguous().view(ebsize, -1)).view(bsize, nsent, lo, -1)
				loss = lossf(output, ot).sum(-1).view(bsize, -1).sum(-1)
			if norm_token:
				lenv = ot.ne(pad_id).int().view(bsize, -1).sum(-1).to(loss)
				loss = loss / lenv
			f.write("\n".join([str(rsu) for rsu in loss.tolist()]).encode("utf-8"))
			loss = output = ot = seq_batch = seq_o = None
			f.write(ens)

td.close()
