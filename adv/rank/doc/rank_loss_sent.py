#encoding: utf-8

# usage: python rank.py rsf h5f models...

norm_token = True

import sys
import torch

from loss.base import LabelSmoothingLoss
from parallel.base import DataParallelCriterion
from parallel.parallelMT import DataParallelMT
from transformer.EnsembleNMT import NMT as Ensemble
from transformer.NMT import NMT
from utils.base import set_random_seed
from utils.fmt.base import sys_open
from utils.fmt.base4torch import parse_cuda
from utils.h5serial import h5File
from utils.io import load_model_cpu
from utils.torch.comp import torch_autocast, torch_compile, torch_inference_mode
from utils.tqdm import tqdm

import cnfg.base as cnfg
from cnfg.ihyp import *
from cnfg.vocab.base import pad_id

def load_fixing(module):

	if hasattr(module, "fix_load"):
		module.fix_load()

td = h5File(sys.argv[2], "r")

ntest = td["ndata"][()].item()
nword = td["nword"][()].tolist()
nwordi, nwordt = nword[0], nword[-1]

if len(sys.argv) == 4:
	mymodel = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.act_drop, cnfg.share_emb, cnfg.nhead, cache_len_default, cnfg.attn_hsize, cnfg.norm_output, cnfg.bindDecoderEmb, cnfg.forbidden_indexes)

	mymodel = load_model_cpu(sys.argv[3], mymodel)
	mymodel.apply(load_fixing)

else:
	models = []
	for modelf in sys.argv[3:]:
		tmp = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.act_drop, cnfg.share_emb, cnfg.nhead, cache_len_default, cnfg.attn_hsize, cnfg.norm_output, cnfg.bindDecoderEmb, cnfg.forbidden_indexes)

		tmp = load_model_cpu(modelf, tmp)
		tmp.apply(load_fixing)

		models.append(tmp)
	mymodel = Ensemble(models)

mymodel.eval()

lossf = LabelSmoothingLoss(nwordt, cnfg.label_smoothing, ignore_index=pad_id, reduction="none", forbidden_index=cnfg.forbidden_indexes)

use_cuda, cuda_device, cuda_devices, multi_gpu = parse_cuda(cnfg.use_cuda, cnfg.gpuid)
use_amp = cnfg.use_amp and use_cuda

# Important to make cudnn methods deterministic
set_random_seed(cnfg.seed, use_cuda)

if cuda_device:
	mymodel.to(cuda_device, non_blocking=True)
	lossf.to(cuda_device, non_blocking=True)
	if multi_gpu:
		mymodel = DataParallelMT(mymodel, device_ids=cuda_devices, output_device=cuda_device.index, host_replicate=True, gather_output=False)
		lossf = DataParallelCriterion(lossf, device_ids=cuda_devices, output_device=cuda_device.index, replicate_once=True)

mymodel = torch_compile(mymodel, *torch_compile_args, **torch_compile_kwargs)
lossf = torch_compile(lossf, *torch_compile_args, **torch_compile_kwargs)

ens = "\n".encode("utf-8")

src_grp, tgt_grp = td["src"]["4"], td["tgt"]["4"]
with sys_open(sys.argv[1], "wb") as f, torch_inference_mode():
	for i in tqdm(range(ntest), mininterval=tqdm_mininterval):
		_curid = str(i)
		seq_batch = torch.from_numpy(src_grp[_curid][()])
		seq_o = torch.from_numpy(tgt_grp[_curid][()])
		bsize, nsent = seq_batch.size()[:2]
		ebsize = bsize * nsent
		if cuda_device:
			seq_batch = seq_batch.to(cuda_device, non_blocking=True)
			seq_o = seq_o.to(cuda_device, non_blocking=True)
		seq_batch, seq_o = seq_batch.long(), seq_o.long()
		lo = seq_o.size(-1) - 1
		ot = seq_o.narrow(-1, 1, lo).contiguous()
		with torch_autocast(enabled=use_amp):
			output = mymodel(seq_batch.view(ebsize, -1), seq_o.narrow(-1, 0, lo).contiguous().view(ebsize, -1)).view(bsize, nsent, lo, -1)
			loss = lossf(output, ot).view(bsize, -1).sum(-1)
		if norm_token:
			lenv = ot.ne(pad_id).int().view(bsize, -1).sum(-1).to(loss, non_blocking=True)
			loss = loss / lenv
		f.write("\n".join([str(rsu) for rsu in loss.tolist()]).encode("utf-8"))
		f.write(ens)
		loss = output = ot = seq_batch = seq_o = None

td.close()
