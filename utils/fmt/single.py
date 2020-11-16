#encoding: utf-8

from utils.fmt.base import list_reader, get_bsize, map_batch, pad_batch

def batch_loader(finput, bsize, maxpad, maxpart, maxtoken, minbsize):

	rsi = []
	nd = maxlen = minlen = mlen_i = 0
	_bsize = bsize
	for i_d in list_reader(finput):
		lgth = len(i_d)
		if maxlen == 0:
			_maxpad = max(1, min(maxpad, lgth // maxpart + 1) // 2)
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
			_maxpad = max(1, min(maxpad, lgth // maxpart + 1) // 2)
			maxlen = lgth + _maxpad
			minlen = lgth - _maxpad
			_bsize = get_bsize(maxlen, maxtoken, bsize)
			nd = 1
	if rsi:
		yield rsi, mlen_i

def batch_mapper(finput, vocabi, bsize, maxpad, maxpart, maxtoken, minbsize):

	for i_d, mlen_i in batch_loader(finput, bsize, maxpad, maxpart, maxtoken, minbsize):
		rsi, extok_i = map_batch(i_d, vocabi)
		yield rsi, mlen_i + extok_i

def batch_padder(finput, vocabi, bsize, maxpad, maxpart, maxtoken, minbsize):

	for i_d, mlen_i in batch_mapper(finput, vocabi, bsize, maxpad, maxpart, maxtoken, minbsize):
		yield pad_batch(i_d, mlen_i)
