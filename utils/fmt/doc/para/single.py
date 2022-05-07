#encoding: utf-8

from utils.fmt.base import map_batch
from utils.fmt.doc.mono.single import batch_loader, batch_padder as batch_padder_base

def batch_mapper(finput, vocabi, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=None):

	_batch_loader = batch_loader if custom_batch_loader is None else custom_batch_loader
	for i_d, mlen_i, nsent in _batch_loader(finput, bsize, maxpad, maxpart, maxtoken, minbsize):
		rsi, extok_i = map_batch(i_d, vocabi)
		yield rsi, mlen_i + extok_i, nsent

def batch_padder(finput, vocabi, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=None, custom_batch_mapper=None):

	return batch_padder_base(finput, vocabi, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=custom_batch_loader, custom_batch_mapper=batch_mapper if custom_batch_mapper is None else custom_batch_mapper)
