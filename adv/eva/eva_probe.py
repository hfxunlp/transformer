#encoding: utf-8

import sys
import torch

from loss.base import LabelSmoothingLoss
from parallel.base import DataParallelCriterion
from parallel.parallelMT import DataParallelMT
from transformer.Probe.NMT import NMT
from utils.base import set_random_seed
from utils.fmt.base4torch import parse_cuda
from utils.h5serial import h5File
from utils.io import load_model_cpu
from utils.torch.comp import torch_autocast, torch_compile, torch_inference_mode
from utils.tqdm import tqdm

import cnfg.probe as cnfg
from cnfg.ihyp import *
from cnfg.vocab.base import pad_id

probe_reorder = cnfg.probe_reorder

def load_fixing(module):

	if hasattr(module, "fix_load"):
		module.fix_load()

def eva(ed, nd, model, lossf, mv_device, multi_gpu, use_amp=False):

	global probe_reorder
	ind_shift = 2 if probe_reorder else 1

	r = 0
	w = 0
	sum_loss = 0.0
	model.eval()
	src_grp, tgt_grp = ed["src"], ed["tgt"]
	with torch_inference_mode():
		for i in tqdm(range(nd), mininterval=tqdm_mininterval):
			bid = str(i)
			seq_batch = torch.from_numpy(src_grp[bid][()])
			seq_o = torch.from_numpy(tgt_grp[bid][()])
			lo = seq_o.size(1) - ind_shift
			if mv_device:
				seq_batch = seq_batch.to(mv_device, non_blocking=True)
				seq_o = seq_o.to(mv_device, non_blocking=True)
			seq_batch, seq_o = seq_batch.long(), seq_o.long()
			ot = seq_o.narrow(1, ind_shift, lo).contiguous()
			with torch_autocast(enabled=use_amp):
				output = model(seq_batch, seq_o.narrow(1, 0, lo))
				loss = lossf(output, ot)
				if multi_gpu:
					loss = loss.sum()
					trans = torch.cat([outu.argmax(-1).to(mv_device, non_blocking=True) for outu in output], 0)
				else:
					trans = output.argmax(-1)
			sum_loss += loss.data.item()
			data_mask = ot.ne(0)
			correct = (trans.eq(ot) & data_mask).int()
			w += data_mask.int().sum().item()
			r += correct.sum().item()
			correct = data_mask = trans = loss = output = ot = seq_batch = seq_o = None
	w = float(w)
	return sum_loss / w, (w - r) / w * 100.0

td = h5File(sys.argv[1], "r")

ntest = td["ndata"][()].item()
nword = td["nword"][()].tolist()
nwordi, nwordt = nword[0], nword[-1]

mymodel = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.act_drop, cnfg.share_emb, cnfg.nhead, cache_len_default, cnfg.attn_hsize, cnfg.norm_output, cnfg.bindDecoderEmb, cnfg.forbidden_indexes, cnfg.num_layer_fwd)

mymodel = load_model_cpu(sys.argv[2], mymodel)
mymodel.apply(load_fixing)

if cnfg.probe_remove_self:
	mymodel.dec.nets[-1].perform_self_attn = False
elif cnfg.probe_remove_cross:
	mymodel.dec.nets[-1].perform_cross_attn = False

mymodel.eval()

lossf = LabelSmoothingLoss(nwordt, cnfg.label_smoothing, ignore_index=pad_id, reduction="sum", forbidden_index=cnfg.forbidden_indexes)

use_cuda = cnfg.use_cuda
gpuid = cnfg.gpuid

use_cuda, cuda_device, cuda_devices, multi_gpu = parse_cuda(cnfg.use_cuda, cnfg.gpuid)

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

use_amp = cnfg.use_amp and use_cuda

vloss, vprec = eva(td, ntest, mymodel, lossf, cuda_device, multi_gpu, use_amp)

td.close()

print("loss/error: %.3f %.2f" % (vloss, vprec,))
