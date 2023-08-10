#encoding: utf-8

from utils.fmt.doc.mono.single import batch_padder as batch_padder_base
from utils.fmt.vocab.base import map_batch

def batch_padder(finput, vocabi, bsize, maxpad, maxpart, maxtoken, minbsize, map_batch=map_batch, **kwargs):

	return batch_padder_base(finput, vocabi, bsize, maxpad, maxpart, maxtoken, minbsize, map_batch=map_batch, **kwargs)
