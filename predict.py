#encoding: utf-8

import sys

import torch

from tqdm import tqdm

import h5py

import cnfg

from transformer.NMT import NMT
from transformer.EnsembleNMT import NMT as Ensemble
from parallel.parallelMT import DataParallelMT

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

def ldvocab(vfile, minf = False, omit_vsize = False):

	global has_unk

	if has_unk:
		rs = {"<pad>":0, "<sos>":1, "<eos>":2, "<unk>":3}
		cwd = 4
	else:
		rs = {"<pad>":0, "<sos>":1, "<eos>":2}
		cwd = 3
	vsize = False
	if omit_vsize:
		vsize = omit_vsize
	for data in list_reader(vfile):
		freq = int(data[0])
		if (not minf) or freq > minf:
			if vsize:
				ndata = len(data) - 1
				if vsize >= ndata:
					for wd in data[1:]:
						rs[wd] = cwd
						cwd += 1
				else:
					for wd in data[1:vsize + 1]:
						rs[wd] = cwd
						cwd += 1
						ndata = vsize
					break
				vsize -= ndata
				if vsize <= 0:
					break
			else:
				for wd in data[1:]:
					rs[wd] = cwd
					cwd += 1
		else:
			break

	return rs, cwd

def reverse_dict(din):

	rs = {}
	for k, v in din.items():
		rs[v] = k

	return rs

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

td = h5py.File(cnfg.test_data, "r")

ntest = int(td["ndata"][:][0])
nwordi = int(td["nwordi"][:][0])
vcbt, nwordt = ldvocab(sys.argv[2])
vcbt = reverse_dict(vcbt)

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
	if multi_gpu:
		mymodel = DataParallelMT(mymodel, device_ids=cuda_devices, output_device=cuda_device.index, host_replicate=True, gather_output=False)

beam_size = cnfg.beam_size

length_penalty = cnfg.length_penalty

ens = "\n".encode("utf-8")

with open(sys.argv[1], "wb") as f:
	with torch.no_grad():
		for i in tqdm(range(ntest)):
			seq_batch = torch.from_numpy(td["i" + str(i)][:]).long()
			if use_cuda:
				seq_batch = seq_batch.to(cuda_device)
			output = mymodel.decode(seq_batch, beam_size, None, length_penalty)
			#output = mymodel.train_decode(seq_batch, beam_size, None, length_penalty)
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
					if (tmpu == 2) or (tmpu == 0):
						break
					else:
						tmp.append(vcbt[tmpu])
				f.write(" ".join(tmp).encode("utf-8"))
				f.write(ens)

td.close()
