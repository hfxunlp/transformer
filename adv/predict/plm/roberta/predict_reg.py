#encoding: utf-8

import sys
import torch
from torch import nn

from parallel.parallelMT import DataParallelMT
from transformer.EnsembleNMT import NMT as Ensemble
from transformer.Prompt.RoBERTa.NMT import NMT
from utils.base import set_random_seed
from utils.fmt.base import sys_open
from utils.fmt.base4torch import parse_cuda_decode
from utils.fmt.plm.base import fix_parameter_name
from utils.func import identity_func
from utils.h5serial import h5File
from utils.io import load_model_cpu
from utils.torch.comp import torch_autocast, torch_compile, torch_inference_mode
from utils.tqdm import tqdm

import cnfg.prompt.roberta.base as cnfg
from cnfg.prompt.roberta.ihyp import *
from cnfg.vocab.plm.roberta import vocab_size

reg_weight, reg_bias = cnfg.reg_weight, cnfg.reg_bias
if reg_weight is None:
	scale_func = identity_func if reg_bias is None else (lambda x: x - reg_bias)
else:
	scale_func = (lambda x: x / reg_weight) if reg_bias is None else (lambda x: x / reg_weight - reg_bias)

def init_fixing(module):

	if hasattr(module, "fix_init"):
		module.fix_init()

def load_fixing(module):

	if hasattr(module, "fix_load"):
		module.fix_load()

nwordi = nwordt = vocab_size

pre_trained_m = cnfg.pre_trained_m
_num_args = len(sys.argv)
if _num_args == 3:
	mymodel = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, fhsize=cnfg.ff_hsize, dropout=cnfg.drop, attn_drop=cnfg.attn_drop, act_drop=cnfg.act_drop, global_emb=cnfg.share_emb, num_head=cnfg.nhead, xseql=cache_len_default, ahsize=cnfg.attn_hsize, norm_output=cnfg.norm_output, bindDecoderEmb=cnfg.bindDecoderEmb, forbidden_index=cnfg.forbidden_indexes, model_name=cnfg.model_name)
	mymodel.dec.lsm = nn.Softmax(-1)
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
else:
	models = []
	for modelf in sys.argv[2:]:
		tmp = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, fhsize=cnfg.ff_hsize, dropout=cnfg.drop, attn_drop=cnfg.attn_drop, act_drop=cnfg.act_drop, global_emb=cnfg.share_emb, num_head=cnfg.nhead, xseql=cache_len_default, ahsize=cnfg.attn_hsize, norm_output=cnfg.norm_output, bindDecoderEmb=cnfg.bindDecoderEmb, forbidden_index=cnfg.forbidden_indexes, model_name=cnfg.model_name)
		tmp.dec.lsm = nn.Softmax(-1)
		if pre_trained_m is not None:
			print("Load pre-trained model from: " + pre_trained_m)
			mymodel.load_plm(fix_parameter_name(torch.load(pre_trained_m, map_location="cpu")))
		if (cnfg.classifier_indices is not None) and hasattr(mymodel, "update_classifier"):
			print("Build new classifier")
			mymodel.update_classifier(torch.as_tensor(cnfg.classifier_indices, dtype=torch.long))
		print("Load pre-trained model from: " + modelf)
		tmp = load_model_cpu(modelf, tmp)
		tmp.apply(load_fixing)
		models.append(tmp)
	mymodel = Ensemble(models)

mymodel.eval()

use_cuda, cuda_device, cuda_devices, multi_gpu = parse_cuda_decode(cnfg.use_cuda, cnfg.gpuid, cnfg.multi_gpu_decoding)
use_amp = cnfg.use_amp and use_cuda

set_random_seed(cnfg.seed, use_cuda)

if cuda_device:
	mymodel.to(cuda_device, non_blocking=True)
	if multi_gpu:
		mymodel = DataParallelMT(mymodel, device_ids=cuda_devices, output_device=cuda_device.index, host_replicate=True, gather_output=False)

mymodel = torch_compile(mymodel, *torch_compile_args, **torch_compile_kwargs)

ens = "\n".encode("utf-8")
with sys_open(sys.argv[1], "wb") as f, h5File(cnfg.test_data, "r") as td, torch_inference_mode():
	src_grp = td["src"]
	for i in tqdm(range(td["ndata"][()].item()), mininterval=tqdm_mininterval):
		seq_batch = torch.from_numpy(src_grp[str(i)][()])
		if cuda_device:
			seq_batch = seq_batch.to(cuda_device, non_blocking=True)
		seq_batch = seq_batch.long()
		with torch_autocast(enabled=use_amp):
			output = mymodel(seq_batch)
		if multi_gpu:
			tmp = []
			for ou in output:
				tmp.extend(scale_func(ou.select(-1, -1)).tolist())
			output = tmp
		else:
			output = scale_func(output.select(-1, -1)).tolist()
		f.write("\n".join([str(_) for _ in output]).encode("utf-8"))
		f.write(ens)
