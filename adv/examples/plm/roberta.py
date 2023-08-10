#encoding: utf-8

import torch
from transformers import RobertaModel

from transformer.PLM.RoBERTa.NMT import NMT
from utils.fmt.plm.base import fix_parameter_name
from utils.torch.comp import torch_inference_mode

import cnfg.plm.roberta.base as cnfg
from cnfg.plm.roberta.ihyp import *
from cnfg.vocab.plm.roberta import vocab_size

def init_fixing(module):

	if hasattr(module, "fix_init"):
		module.fix_init()

print("load pre-trained models")

tmod = NMT(cnfg.isize, vocab_size, vocab_size, cnfg.nlayer, fhsize=cnfg.ff_hsize, dropout=cnfg.drop, attn_drop=cnfg.attn_drop, act_drop=cnfg.act_drop, global_emb=cnfg.share_emb, num_head=cnfg.nhead, xseql=cache_len_default, ahsize=cnfg.attn_hsize, norm_output=cnfg.norm_output, bindDecoderEmb=cnfg.bindDecoderEmb, forbidden_index=cnfg.forbidden_indexes, model_name=cnfg.model_name)
tmod.apply(init_fixing)
tmod.load_plm(fix_parameter_name(torch.load("plm/roberta-base/pytorch_model.bin", map_location="cpu")))

tmod.eval()

print("load models with transformers")
smod = RobertaModel.from_pretrained("plm/roberta-base/")
smod.eval()

print("forward with transformers")
td = torch.as_tensor([0, 100, 33, 41, 15162, 4, 2], dtype=torch.long).unsqueeze(0)

with torch_inference_mode():
	ers = smod(td).last_hidden_state
	print("forward for test")
	trs = tmod(td)
print(ers)
print(trs)
