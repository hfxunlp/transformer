#encoding: utf-8

""" usage:
	python eva.py $input_file.h5 k $rs_file.h5 $model_file.h5 ...
"""

import sys

from numpy import array as np_array, int32 as np_int32

import torch
from torch import nn

from utils.tqdm import tqdm

from utils.h5serial import h5File

import cnfg.base as cnfg
from cnfg.ihyp import *

from transformer.NMT import NMT
from transformer.EnsembleNMT import NMT as Ensemble
from parallel.parallelMT import DataParallelMT

from utils.base import *
from utils.fmt.base import pad_id
from utils.fmt.base4torch import parse_cuda_decode

def load_fixing(module):

	if hasattr(module, "fix_load"):
		module.fix_load()

k = int(sys.argv[2])

td = h5File(sys.argv[1], "r")

ntest = td["ndata"][()].item()
nword = td["nword"][()].tolist()
nwordi, nwordt = nword[0], nword[-1]

if len(sys.argv) == 5:
	mymodel = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.share_emb, cnfg.nhead, cache_len_default, cnfg.attn_hsize, cnfg.norm_output, cnfg.bindDecoderEmb, cnfg.forbidden_indexes)

	mymodel = load_model_cpu(sys.argv[4], mymodel)
	mymodel.apply(load_fixing)
	mymodel.dec.lsm = nn.Softmax(-1)

else:
	models = []
	for modelf in sys.argv[4:]:
		tmp = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.share_emb, cnfg.nhead, cache_len_default, cnfg.attn_hsize, cnfg.norm_output, cnfg.bindDecoderEmb, cnfg.forbidden_indexes)

		tmp = load_model_cpu(modelf, tmp)
		tmp.apply(load_fixing)
		# dec.lsm is NOT used by Ensemble
		tmp.dec.lsm = nn.Softmax(-1)

		models.append(tmp)
	mymodel = Ensemble(models)

mymodel.eval()

use_cuda, cuda_device, cuda_devices, multi_gpu = parse_cuda_decode(cnfg.use_cuda, cnfg.gpuid, cnfg.multi_gpu_decoding)
use_amp = cnfg.use_amp and use_cuda

set_random_seed(cnfg.seed, use_cuda)

if cuda_device:
	mymodel.to(cuda_device)
	if multi_gpu:
		mymodel = DataParallelMT(mymodel, device_ids=cuda_devices, output_device=cuda_device.index, host_replicate=True, gather_output=True)

beam_size = cnfg.beam_size
length_penalty = cnfg.length_penalty

src_grp, tgt_grp = td["src"], td["tgt"]
with h5File(sys.argv[3], "w", libver=h5_libver) as rsf, torch.no_grad():
	p_grp = rsf.create_group("p")
	ind_grp = rsf.create_group("ind")
	for i in tqdm(range(ntest), mininterval=tqdm_mininterval):
		bid = str(i)
		seq_batch = torch.from_numpy(src_grp[bid][()])
		seq_o = torch.from_numpy(tgt_grp[bid][()])
		lo = seq_o.size(1) - 1
		if cuda_device:
			seq_batch = seq_batch.to(cuda_device)
			seq_o = seq_o.to(cuda_device)
		seq_batch, seq_o = seq_batch.long(), seq_o.long()
		with autocast(enabled=use_amp):
			output = mymodel(seq_batch, seq_o.narrow(1, 0, lo))
		p, ind = output.masked_fill_(seq_o.narrow(1, 1, lo).eq(pad_id).unsqueeze(-1), 0.0).topk(k, dim=-1)
		ind = ind.int()
		if cuda_device:
			p = p.cpu()
			ind = ind.cpu()
		p_grp.create_dataset(bid, data=p, **h5datawargs)
		ind_grp.create_dataset(bid, data=ind, **h5datawargs)
	rsf["ndata"] = np_array([ntest], dtype=np_int32)

td.close()
