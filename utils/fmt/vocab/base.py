#encoding: utf-8

from cnfg.vocab.base import eos_id, sos_id, unk_id, use_unk

def reverse_dict(din):

	return {v:k for k, v in din.items()}

def merge_vocab(*vcbin):

	rs = {}
	for _ in vcbin:
		for k, v in _.items():
			rs[k] = rs.get(k, 0) + v

	return rs

def legal_vocab(sent, ilgset, ratio):

	total = ilg = 0
	for tmpu in sent.split():
		if tmpu:
			if tmpu in ilgset:
				ilg += 1
			total += 1
	rt = float(ilg) / float(total)

	return rt < ratio

def no_unk_mapper(vcb, ltm, print_func=None):

	if print_func is None:
		return [vcb[wd] for wd in ltm if wd in vcb]
	else:
		rs = []
		for wd in ltm:
			if wd in vcb:
				rs.append(vcb[wd])
			else:
				print_func("Error mapping: "+ wd)
		return rs

def map_instance(i_d, vocabi, use_unk=use_unk, sos_id=sos_id, eos_id=eos_id, unk_id=unk_id, **kwargs):

	rsi = [sos_id]
	rsi.extend([vocabi.get(wd, unk_id) for wd in i_d] if use_unk else no_unk_mapper(vocabi, i_d))#[vocabi[wd] for wd in i_d if wd in vocabi]
	rsi.append(eos_id)

	return rsi

def map_batch_core(i_d, vocabi, use_unk=use_unk, sos_id=sos_id, eos_id=eos_id, unk_id=unk_id, **kwargs):

	if isinstance(i_d[0], (tuple, list,)):
		return [map_batch_core(idu, vocabi, use_unk=use_unk, sos_id=sos_id, eos_id=eos_id, unk_id=unk_id, **kwargs) for idu in i_d]
	else:
		return map_instance(i_d, vocabi, use_unk=use_unk, sos_id=sos_id, eos_id=eos_id, unk_id=unk_id, **kwargs)

def map_batch(i_d, vocabi, use_unk=use_unk, sos_id=sos_id, eos_id=eos_id, unk_id=unk_id, **kwargs):

	return map_batch_core(i_d, vocabi, use_unk=use_unk, sos_id=sos_id, eos_id=eos_id, unk_id=unk_id, **kwargs), 2
