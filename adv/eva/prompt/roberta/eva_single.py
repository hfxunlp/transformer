#encoding: utf-8

import sys
import torch

from loss.base import NLLLoss
from parallel.base import DataParallelCriterion
from parallel.parallelMT import DataParallelMT
from transformer.Prompt.RoBERTa.NMT import NMT
from utils.base import set_random_seed
from utils.fmt.base4torch import parse_cuda
from utils.fmt.plm.base import fix_parameter_name
from utils.h5serial import h5File
from utils.io import load_model_cpu
from utils.torch.comp import torch_autocast, torch_compile, torch_inference_mode
from utils.tqdm import tqdm

import cnfg.prompt.roberta.base as cnfg
from cnfg.prompt.roberta.ihyp import *
from cnfg.vocab.plm.roberta import vocab_size

def load_fixing(module):

	if hasattr(module, "fix_load"):
		module.fix_load()

def eva(ed, nd, model, lossf, mv_device, multi_gpu, use_amp=False):
	r = w = 0
	sum_loss = 0.0
	model.eval()
	src_grp, tgt_grp = ed["src"], ed["tgt"]
	with torch_inference_mode():
		for i in tqdm(range(nd), mininterval=tqdm_mininterval):
			bid = str(i)
			seq_batch = torch.from_numpy(src_grp[bid][()])
			seq_o = torch.from_numpy(tgt_grp[bid][()]).squeeze(-1)
			if mv_device:
				seq_batch = seq_batch.to(mv_device, non_blocking=True)
				seq_o = seq_o.to(mv_device, non_blocking=True)
			seq_batch, seq_o = seq_batch.long(), seq_o.long()
			with torch_autocast(enabled=use_amp):
				output = model(seq_batch)
				loss = lossf(output, seq_o)
				if multi_gpu:
					loss = loss.sum()
					trans = torch.cat([outu.argmax(-1).to(mv_device, non_blocking=True) for outu in output], 0)
				else:
					trans = output.argmax(-1)
			sum_loss += loss.data.item()
			w += seq_o.numel()
			r += trans.eq(seq_o).int().sum().item()
			trans = loss = output = seq_batch = seq_o = None
	w = float(w)
	return sum_loss / w, (w - r) / w * 100.0

nwordi = nwordt = vocab_size

mymodel = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, fhsize=cnfg.ff_hsize, dropout=cnfg.drop, attn_drop=cnfg.attn_drop, act_drop=cnfg.act_drop, global_emb=cnfg.share_emb, num_head=cnfg.nhead, xseql=cache_len_default, ahsize=cnfg.attn_hsize, norm_output=cnfg.norm_output, bindDecoderEmb=cnfg.bindDecoderEmb, forbidden_index=cnfg.forbidden_indexes, model_name=cnfg.model_name)

# important to load the pre-trained model, as the load_plm function not only load parameters, but also may introduce new parameters, which affects the parameter alignment.
pre_trained_m = cnfg.pre_trained_m
if pre_trained_m is not None:
	print("Load pre-trained model from: " + pre_trained_m)
	mymodel.load_plm(fix_parameter_name(torch.load(pre_trained_m, map_location="cpu")))
if (cnfg.classifier_indices is not None) and hasattr(mymodel, "update_classifier"):
	print("Build new classifier")
	mymodel.update_classifier(torch.as_tensor(cnfg.classifier_indices, dtype=torch.long))
fine_tune_m = sys.argv[2]
print("Load pre-trained model from: " + fine_tune_m)
mymodel = load_model_cpu(fine_tune_m, mymodel)
mymodel.apply(load_fixing)

mymodel.eval()

lossf = NLLLoss(reduction="sum")

use_cuda = cnfg.use_cuda
gpuid = cnfg.gpuid

use_cuda, cuda_device, cuda_devices, multi_gpu = parse_cuda(cnfg.use_cuda, cnfg.gpuid)

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

with h5File(sys.argv[1], "r") as td:
	vloss, vprec = eva(td, td["ndata"][()].item(), mymodel, lossf, cuda_device, multi_gpu, use_amp)

print("loss/error: %.3f %.2f" % (vloss, vprec,))
