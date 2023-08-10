#encoding: utf-8

import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast as Tokenizer

from transformer.PLM.T5.NMT import NMT
from utils.fmt.plm.base import fix_parameter_name
from utils.torch.comp import torch_inference_mode

import cnfg.plm.t5.base as cnfg
from cnfg.plm.t5.ihyp import *
from cnfg.vocab.plm.t5 import vocab_size

def init_fixing(module):

	if hasattr(module, "fix_init"):
		module.fix_init()

print("load pre-trained models")
tokenizer = Tokenizer(tokenizer_file="plm/t5-base/tokenizer.json")

tmod = NMT(cnfg.isize, vocab_size, vocab_size, cnfg.nlayer, fhsize=cnfg.ff_hsize, dropout=cnfg.drop, attn_drop=cnfg.attn_drop, act_drop=cnfg.act_drop, global_emb=cnfg.share_emb, num_head=cnfg.nhead, xseql=cache_len_default, ahsize=cnfg.attn_hsize, norm_output=cnfg.norm_output, bindDecoderEmb=cnfg.bindDecoderEmb, forbidden_index=cnfg.forbidden_indexes, model_name=cnfg.model_name)
tmod.apply(init_fixing)
tmod.load_plm(fix_parameter_name(torch.load("plm/t5-base/pytorch_model.bin", map_location="cpu")))

tmod.eval()

print("load models with transformers")
smod = T5ForConditionalGeneration.from_pretrained("plm/t5-base/")
smod.eval()

print("forward with transformers")
tde = torch.as_tensor([27, 43, 192, 16981, 5, 1], dtype=torch.long).unsqueeze(0)
tdo = torch.as_tensor([0, 531, 25, 241, 80, 58], dtype=torch.long).unsqueeze(0)

with torch_inference_mode():
	ers = smod(input_ids=tde, decoder_input_ids=tdo, output_hidden_states=True).decoder_hidden_states[-1]
	print("forward for test")
	trs = tmod(tde, tdo)
print(ers)
print(trs)

tde = torch.as_tensor([27, 43, 32099, 16981, 5, 32098, 241, 80, 58, 1], dtype=torch.long).unsqueeze(0)
with torch_inference_mode():
	ers = smod.generate(tde)
	trs = tmod.decode(tde)
print(tokenizer.convert_ids_to_tokens(ers.squeeze(0)))
print(tokenizer.convert_ids_to_tokens(trs.squeeze(0)))
