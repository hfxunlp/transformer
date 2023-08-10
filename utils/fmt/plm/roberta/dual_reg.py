#encoding: utf-8

from utils.fmt.plm.dual_reg import batch_padder as batch_padder_base

from cnfg.vocab.plm.roberta import pad_id

def batch_padder(finput, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize, pad_id=pad_id, **kwargs):

	return batch_padder_base(finput, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize, pad_id=pad_id, **kwargs)
