#encoding: utf-8

# usage: python rank.py rsf h5f models...

import sys

import torch

from tqdm import tqdm

import h5py

import cnfg

from transformer.NMT import NMT
from transformer.EnsembleNMT import NMT as Ensemble
from parallel.parallelMT import DataParallelMT

from loss import LabelSmoothingLoss

has_unk = True

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

def load_model_cpu_old(modf, base_model):

	base_model.load_state_dict(torch.load(modf, map_location='cpu'))

	return base_model

def load_model_cpu(modf, base_model):

	mpg = torch.load(modf, map_location='cpu')

	for para, mp in zip(base_model.parameters(), mpg):
		para.data = mp.data

	return base_model

def load_fixing(module):

	if "fix_load" in dir(module):
		module.fix_load()

td = h5py.File(sys.argv[2], "r")

ntest = int(td["ndata"][:][0])
nwordi = int(td["nwordi"][:][0])
nwordt = int(td["nwordt"][:][0])

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

if use_cuda:
	mymodel.to(cuda_device)
	lossf.to(cuda_device)
	if multi_gpu:
		mymodel = DataParallelMT(mymodel, device_ids=cuda_devices, output_device=cuda_device.index, host_replicate=True, gather_output=False)
		lossf = DataParallelCriterion(lossf, device_ids=cuda_devices, output_device=cuda_device.index, replicate_once=True)

ens = "\n".encode("utf-8")

with open(sys.argv[1], "wb") as f:
	with torch.no_grad():
		for i in tqdm(range(ntest)):
			seq_batch = torch.from_numpy(td["i" + str(i)][:]).long()
			seq_o = torch.from_numpy(td["t" + str(i)][:]).long()
			if use_cuda:
				seq_batch = seq_batch.to(cuda_device)
				seq_o = seq_o.to(cuda_device)
			lo = seq_o.size(1) - 1
			ot = seq_o.narrow(1, 1, lo).contiguous()
			output = mymodel(seq_batch, seq_o.narrow(1, 0, lo))
			loss = lossf(output, ot)
			lenv = (lo - ot.eq(0).sum(-1)).to(loss)
			loss = loss.sum(-1).view(-1, lo).sum(-1) / lenv
			f.write("\n".join([str(rsu) for rsu in loss.tolist()]).encode("utf-8"))
			f.write(ens)

td.close()

