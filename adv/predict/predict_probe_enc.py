#encoding: utf-8

# usage: python file_name.py test.eva.h5 model.h5 tgt.vcb $rsf.txt

import sys

import torch
from torch.cuda.amp import autocast

from tqdm import tqdm

import h5py

import cnfg.probe as cnfg
from cnfg.ihyp import *

from transformer.Probe.NMT import NMT

from utils.base import *
from utils.fmt.base import ldvocab, reverse_dict, init_vocab, sos_id, eos_id
from utils.fmt.base4torch import parse_cuda_decode

def load_fixing(module):

	if hasattr(module, "fix_load"):
		module.fix_load()

td = h5py.File(sys.argv[1], "r")

ntest = td["ndata"][:].item()
nwordi = td["nword"][:].tolist()[0]
vcbt, nwordt = ldvocab(sys.argv[3])
vcbt = reverse_dict(vcbt)

mymodel = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.share_emb, cnfg.nhead, cache_len_default, cnfg.attn_hsize, cnfg.norm_output, cnfg.bindDecoderEmb, cnfg.forbidden_indexes, cnfg.num_layer_fwd)

mymodel = load_model_cpu(sys.argv[2], mymodel)
mymodel.apply(load_fixing)

mymodel.eval()

enc, trans, classifier = mymodel.enc, mymodel.dec.trans, mymodel.dec.classifier

use_cuda, cuda_device, cuda_devices, multi_gpu = parse_cuda_decode(cnfg.use_cuda, cnfg.gpuid, cnfg.multi_gpu_decoding)
# do not support multi-gpu
multi_gpu, cuda_devices = False, None
use_amp = cnfg.use_amp and use_cuda

set_random_seed(cnfg.seed, use_cuda)

if cuda_device:
	enc.to(cuda_device)
	trans.to(cuda_device)
	classifier.to(cuda_device)

ignore_ids = set(init_vocab.values())

src_grp = td["src"]
ens = "\n".encode("utf-8")
with open(sys.argv[4], "wb") as fwrt:
	with torch.no_grad():
		for i in tqdm(range(ntest)):
			bid = str(i)
			seq_batch = torch.from_numpy(src_grp[bid][:])
			if cuda_device:
				seq_batch = seq_batch.to(cuda_device)
			seq_batch = seq_batch.long()
			_mask = seq_batch.eq(0)
			with autocast(enabled=use_amp):
			# mask pad/sos/eos_id in output
				output = classifier(trans(enc(seq_batch, mask=_mask.unsqueeze(1), no_std_out=True))).argmax(-1).masked_fill(_mask | seq_batch.eq(sos_id) | seq_batch.eq(eos_id), 0).tolist()
			for tran in output:
				fwrt.write(" ".join([vcbt[tmpu] for tmpu in tran[1:] if tmpu not in ignore_ids]).encode("utf-8"))
				fwrt.write(ens)

td.close()
