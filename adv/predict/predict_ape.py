#encoding: utf-8

import sys

import torch
from torch.cuda.amp import autocast

from tqdm import tqdm

import h5py

import cnfg.base as cnfg
from cnfg.ihyp import *

from transformer.APE.NMT import NMT
from transformer.EnsembleNMT import NMT as Ensemble
from parallel.parallelMT import DataParallelMT

from utils.base import *
from utils.fmt.base import ldvocab, reverse_dict, eos_id
from utils.fmt.base4torch import parse_cuda_decode

def load_fixing(module):

	if hasattr(module, "fix_load"):
		module.fix_load()

td = h5py.File(cnfg.test_data, "r")

ntest = td["ndata"][:].item()
nwordi = td["nword"][:].tolist()[0]
vcbt, nwordt = ldvocab(sys.argv[2])
vcbt = reverse_dict(vcbt)

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

use_cuda, cuda_device, cuda_devices, multi_gpu = parse_cuda_decode(cnfg.use_cuda, cnfg.gpuid, cnfg.multi_gpu_decoding)

use_amp = cnfg.use_amp and use_cuda

set_random_seed(cnfg.seed, use_cuda)

if cuda_device:
	mymodel.to(cuda_device)
	if multi_gpu:
		mymodel = DataParallelMT(mymodel, device_ids=cuda_devices, output_device=cuda_device.index, host_replicate=True, gather_output=False)

beam_size = cnfg.beam_size

length_penalty = cnfg.length_penalty

ens = "\n".encode("utf-8")

# using tgt instead of mt since data are processed by tools/mkiodata.py for the mt task
src_grp, mt_grp = td["src"], td["tgt"]
with open(sys.argv[1], "wb") as f:
	with torch.no_grad():
		for i in tqdm(range(ntest)):
			seq_batch = torch.from_numpy(src_grp[str(i)][:])
			seq_mt = torch.from_numpy(mt_grp[str(i)][:])
			if cuda_device:
				seq_batch = seq_batch.to(cuda_device)
				seq_mt = seq_mt.to(cuda_device)
			seq_batch, seq_mt = seq_batch.long(), seq_mt.long()
			with autocast(enabled=use_amp):
				output = mymodel.decode(seq_batch, seq_mt, beam_size, None, length_penalty)
			if multi_gpu:
				tmp = []
				for ou in output:
					tmp.extend(ou.tolist())
				output = tmp
			else:
				output = output.tolist()
			for tran in output:
				tmp = []
				for tmpu in tran:
					if tmpu == eos_id:
						break
					else:
						tmp.append(vcbt[tmpu])
				f.write(" ".join(tmp).encode("utf-8"))
				f.write(ens)

td.close()
