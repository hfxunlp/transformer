#encoding: utf-8

import sys
import torch

from parallel.parallelMT import DataParallelMT
from transformer.EnsembleNMT import NMT as Ensemble
from transformer.NMT import NMT
from utils.h5serial import h5File
from utils.io import load_model_cpu
from utils.torch.comp import torch_compile, torch_inference_mode
from utils.tqdm import tqdm

import cnfg.base as cnfg
from cnfg.ihyp import *

def load_fixing(module):

	if hasattr(module, "fix_load"):
		module.fix_load()

td = h5File(cnfg.dev_data, "r")

ntest = td["ndata"][()].item()
nword = td["nword"][()].tolist()
nwordi, nwordt = nword[0], nword[-1]

if len(sys.argv) == 2:
	mymodel = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.act_drop, cnfg.share_emb, cnfg.nhead, cache_len_default, cnfg.attn_hsize, cnfg.norm_output, cnfg.bindDecoderEmb, cnfg.forbidden_indexes)

	mymodel = load_model_cpu(sys.argv[1], mymodel)
	mymodel.apply(load_fixing)

else:
	models = []
	for modelf in sys.argv[1:]:
		tmp = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.act_drop, cnfg.share_emb, cnfg.nhead, cache_len_default, cnfg.attn_hsize, cnfg.norm_output, cnfg.bindDecoderEmb, cnfg.forbidden_indexes)

		tmp = load_model_cpu(modelf, tmp)
		tmp.apply(load_fixing)

		models.append(tmp)
	mymodel = Ensemble(models)

mymodel.eval()

use_cuda = cnfg.use_cuda
gpuid = cnfg.gpuid

if use_cuda and torch.cuda.is_available():
	use_cuda = True
	if len(gpuid.split(",")) > 1:
		if cnfg.multi_gpu_decoding:
			cuda_device = torch.device(gpuid[:gpuid.find(",")].strip())
			cuda_devices = [int(_.strip()) for _ in gpuid[gpuid.find(":") + 1:].split(",")]
			multi_gpu = True
		else:
			cuda_device = torch.device("cuda:" + gpuid[gpuid.rfind(","):].strip())
			multi_gpu = False
			cuda_devices = None
	else:
		cuda_device = torch.device(gpuid)
		multi_gpu = False
		cuda_devices = None
	torch.cuda.set_device(cuda_device.index)
else:
	cuda_device = False
	multi_gpu = False
	cuda_devices = None

if cuda_device:
	mymodel.to(cuda_device, non_blocking=True)
	if multi_gpu:
		mymodel = DataParallelMT(mymodel, device_ids=cuda_devices, output_device=cuda_device.index, host_replicate=True, gather_output=False)

mymodel = torch_compile(mymodel, *torch_compile_args, **torch_compile_kwargs)

beam_size = cnfg.beam_size
length_penalty = cnfg.length_penalty

src_grp = td["src"]
with torch_inference_mode():
	for i in tqdm(range(ntest), mininterval=tqdm_mininterval):
		seq_batch = torch.from_numpy(src_grp[str(i)][()])
		if cuda_device:
			seq_batch = seq_batch.to(cuda_device, non_blocking=True)
		seq_batch = seq_batch.long()
		output = mymodel.decode(seq_batch, beam_size, None, length_penalty)

td.close()
