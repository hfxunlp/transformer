#encoding: utf-8

from utils.fmt.base import reverse_dict
from utils.fmt.json import loadf, dumpf

from cnfg.vocab.plm.roberta import *

def ldvocab(vfile, *args, **kwargs):

	global pad_id, sos_id, eos_id, unk_id, mask_id, vocab_size

	rs = loadf(vfile)

	if "<pad>" in rs:
		pad_id = rs["<pad>"]
	if "<unk>" in rs:
		unk_id = rs["<unk>"]
	if "<s>" in rs:
		sos_id = rs["<s>"]
	if "</s>" in rs:
		eos_id = rs["</s>"]
	if "<mask>" in rs:
		mask_id = rs["<mask>"]
	vocab_size = len(rs)

	return rs, vocab_size

save_vocab = dumpf

def ldvocab_list(vfile, *args, **kwargs):

	_ = reverse_dict(loadf(vfile))
	rs = [_[ind] for ind in sorted(_.keys())]

	return rs, len(rs)
