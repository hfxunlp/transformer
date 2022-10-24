#encoding: utf-8

import sys

from utils.fmt.base import line_reader, reverse_dict

from cnfg.vocab.plm.bert import *

def ldvocab(vfile, *args, **kwargs):

	global pad_id, sos_id, eos_id, unk_id, mask_id, vocab_size

	rs, cwd = {}, 0
	for wd in line_reader(vfile, keep_empty_line=False):
		rs[wd] = cwd
		cwd += 1

	if "[PAD]" in rs:
		pad_id = rs["[PAD]"]
	if "[UNK]" in rs:
		unk_id = rs["[UNK]"]
	if "[CLS]" in rs:
		sos_id = rs["[CLS]"]
	if "[SEP]" in rs:
		eos_id = rs["[SEP]"]
	if "[MASK]" in rs:
		mask_id = rs["[MASK]"]
	vocab_size = cwd

	return rs, cwd

def save_vocab(vcb_dict, fname):

	r_vocab = reverse_dict(vcb_dict)

	freqs = list(r_vocab.keys())
	freqs.sort()

	_ = "\n".join([r_vocab[_key] for _key in freqs])
	with sys.stdout.buffer if fname == "-" else open(fname, "wb") as f:
		f.write(_.encode("utf-8"))
		f.write("\n".encode("utf-8"))

def ldvocab_list(vfile, *args, **kwargs):

	rs = []
	for data in line_reader(vfile, keep_empty_line=False):
		rs.append(data)

	return rs, len(rs)
