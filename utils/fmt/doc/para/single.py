#encoding: utf-8

from utils.fmt.base import map_batch, pad_batch
from utils.fmt.doc.mono.single import batch_loader

def batch_mapper(finput, vocabi, bsize, maxpad, maxpart, maxtoken, minbsize):

	for i_d, mlen_i, nsent in batch_loader(finput, bsize, maxpad, maxpart, maxtoken, minbsize):
		rsi, extok_i = map_batch(i_d, vocabi)
		yield rsi, mlen_i + extok_i, nsent

def batch_padder(finput, vocabi, bsize, maxpad, maxpart, maxtoken, minbsize):

	for i_d, mlen_i, nsent in batch_mapper(finput, vocabi, bsize, maxpad, maxpart, maxtoken, minbsize):
		yield pad_batch(i_d, mlen_i), nsent
