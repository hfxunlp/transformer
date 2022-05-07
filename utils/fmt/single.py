#encoding: utf-8

from utils.fmt.base import list_reader, get_bsize, map_batch, pad_batch
from math import ceil

def batch_loader(finput, bsize, maxpad, maxpart, maxtoken, minbsize):

	_f_maxpart = float(maxpart)
	rsi = []
	nd = maxlen = minlen = mlen_i = 0
	for i_d in list_reader(finput, keep_empty_line=True):
		lgth = len(i_d)
		if maxlen == 0:
			_maxpad = max(1, min(maxpad, ceil(lgth / _f_maxpart)) // 2)
			maxlen = lgth + _maxpad
			minlen = lgth - _maxpad
			_bsize = get_bsize(maxlen, maxtoken, bsize)
		if (nd < minbsize) or (lgth <= maxlen and lgth >= minlen and nd < _bsize):
			rsi.append(i_d)
			if lgth > mlen_i:
				mlen_i = lgth
			nd += 1
		else:
			yield rsi, mlen_i
			rsi = [i_d]
			mlen_i = lgth
			_maxpad = max(1, min(maxpad, ceil(lgth / _f_maxpart)) // 2)
			maxlen = lgth + _maxpad
			minlen = lgth - _maxpad
			_bsize = get_bsize(maxlen, maxtoken, bsize)
			nd = 1
	if rsi:
		yield rsi, mlen_i

def batch_mapper(finput, vocabi, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=None):

	_batch_loader = batch_loader if custom_batch_loader is None else custom_batch_loader
	for i_d, mlen_i in _batch_loader(finput, bsize, maxpad, maxpart, maxtoken, minbsize):
		rsi, extok_i = map_batch(i_d, vocabi)
		yield rsi, mlen_i + extok_i

def batch_padder(finput, vocabi, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=None, custom_batch_mapper=None):

	_batch_mapper = batch_mapper if custom_batch_mapper is None else custom_batch_mapper
	for i_d, mlen_i in _batch_mapper(finput, vocabi, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=custom_batch_loader):
		yield pad_batch(i_d, mlen_i)
