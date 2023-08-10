#encoding: utf-8

import torch
from transformers import MBartForConditionalGeneration, MBartTokenizerFast as Tokenizer

from transformer.PLM.MBART.NMT import NMT
from utils.fmt.plm.base import fix_parameter_name
from utils.torch.comp import torch_inference_mode

import cnfg.plm.mbart.base as cnfg
from cnfg.plm.mbart.ihyp import *
from cnfg.vocab.plm.mbart import vocab_size

def init_fixing(module):

	if hasattr(module, "fix_init"):
		module.fix_init()

print("load pre-trained models")
tokenizer = Tokenizer(tokenizer_file="plm/mbart-large-cc25/tokenizer.json")

tmod = NMT(cnfg.isize, vocab_size, vocab_size, cnfg.nlayer, fhsize=cnfg.ff_hsize, dropout=cnfg.drop, attn_drop=cnfg.attn_drop, act_drop=cnfg.act_drop, global_emb=cnfg.share_emb, num_head=cnfg.nhead, xseql=cache_len_default, ahsize=cnfg.attn_hsize, norm_output=cnfg.norm_output, bindDecoderEmb=cnfg.bindDecoderEmb, forbidden_index=cnfg.forbidden_indexes, model_name=cnfg.model_name)
tmod.apply(init_fixing)
tmod.load_plm(fix_parameter_name(torch.load("plm/mbart-large-cc25/pytorch_model.bin", map_location="cpu")))
tmod.eval()

print("load models with transformers")
smod = MBartForConditionalGeneration.from_pretrained("plm/mbart-large-cc25")
smod.eval()

print("forward with transformers")
tde = torch.as_tensor([17, 765, 142, 108787, 5, 2, 250004], dtype=torch.long).unsqueeze(0)
tdo = torch.as_tensor([250004, 17, 765, 142, 108787, 5, 2], dtype=torch.long).unsqueeze(0)

print("forward for test")
with torch_inference_mode():
	ers = smod(input_ids=tde, decoder_input_ids=tdo, output_hidden_states=True).decoder_hidden_states[-1]
	trs = tmod(tde, tdo)

print(ers)
print(trs)

with torch_inference_mode():
	ers = smod.generate(tde, decoder_start_token_id=250004)
	trs = tmod.decode(tde, lang_id=250004)
print(tokenizer.convert_ids_to_tokens(ers.squeeze(0)))
print(tokenizer.convert_ids_to_tokens(trs.squeeze(0)))
