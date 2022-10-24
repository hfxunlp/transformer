#encoding: utf-8

from cnfg.vocab.plm.bert import pad_id
from utils.fmt.plm.dual import batch_padder as batch_padder_base

def batch_padder(finput, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=None, pad_id=pad_id, **kwargs):

	return batch_padder_base(finput, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=custom_batch_loader, pad_id=pad_id, **kwargs)
