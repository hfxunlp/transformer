#encoding: utf-8

from cnfg.hyp import use_unk

pad_id, sos_id, eos_id = 0, 1, 2
if use_unk:
	unk_id = 3
	init_vocab = {"<pad>":pad_id, "<sos>":sos_id, "<eos>":eos_id, "<unk>":unk_id}
	init_normal_token_id = 4
else:
	unk_id = None
	init_vocab = {"<pad>":pad_id, "<sos>":sos_id, "<eos>":eos_id}
	init_normal_token_id = 3
init_token_id = 3
