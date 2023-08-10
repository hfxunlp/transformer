#encoding: utf-8

# usage: python file_name.py test.eva.h5 model.h5 tgt.vcb $rsf.txt

import sys
import torch

from transformer.Probe.NMT import NMT
from utils.base import set_random_seed
from utils.fmt.base import sys_open
from utils.fmt.base4torch import parse_cuda_decode
from utils.fmt.vocab.base import reverse_dict
from utils.fmt.vocab.token import init_vocab, ldvocab
from utils.h5serial import h5File
from utils.io import load_model_cpu
from utils.torch.comp import torch_autocast, torch_inference_mode
from utils.tqdm import tqdm

import cnfg.probe as cnfg
from cnfg.ihyp import *
from cnfg.vocab.base import eos_id, pad_id, sos_id

def load_fixing(module):

	if hasattr(module, "fix_load"):
		module.fix_load()

td = h5File(sys.argv[1], "r")

ntest = td["ndata"][()].item()
nwordi = td["nword"][()].tolist()[0]
vcbt, nwordt = ldvocab(sys.argv[3])
vcbt = reverse_dict(vcbt)

mymodel = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.act_drop, cnfg.share_emb, cnfg.nhead, cache_len_default, cnfg.attn_hsize, cnfg.norm_output, cnfg.bindDecoderEmb, cnfg.forbidden_indexes, cnfg.num_layer_fwd)

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
	enc.to(cuda_device, non_blocking=True)
	trans.to(cuda_device, non_blocking=True)
	classifier.to(cuda_device, non_blocking=True)

ignore_ids = set(init_vocab.values())

src_grp = td["src"]
ens = "\n".encode("utf-8")
with sys_open(sys.argv[4], "wb") as fwrt, torch_inference_mode():
	for i in tqdm(range(ntest), mininterval=tqdm_mininterval):
		bid = str(i)
		seq_batch = torch.from_numpy(src_grp[bid][()])
		if cuda_device:
			seq_batch = seq_batch.to(cuda_device, non_blocking=True)
		seq_batch = seq_batch.long()
		_mask = seq_batch.eq(pad_id)
		with torch_autocast(enabled=use_amp):
			# mask pad/sos/eos_id in output
			output = classifier(trans(enc(seq_batch, mask=_mask.unsqueeze(1), no_std_out=True))).argmax(-1).masked_fill(_mask | seq_batch.eq(sos_id) | seq_batch.eq(eos_id), 0).tolist()
		for tran in output:
			fwrt.write(" ".join([vcbt[tmpu] for tmpu in tran[1:] if tmpu not in ignore_ids]).encode("utf-8"))
			fwrt.write(ens)

td.close()
