#encoding: utf-8

from math import ceil

from utils.fmt.base import get_bsize, iter_to_int, list_reader as file_reader, pad_batch

from cnfg.vocab.base import pad_id

def batch_loader(finput, bsize, maxpad, maxpart, maxtoken, minbsize, get_bsize=get_bsize, file_reader=file_reader, iter_to_int=iter_to_int, **kwargs):

	_f_maxpart = float(maxpart)
	rsi = []
	nd = maxlen = minlen = mlen_i = 0
	for i_d in file_reader(finput, keep_empty_line=True):
		i_d = list(iter_to_int(i_d))
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

def batch_padder(finput, bsize, maxpad, maxpart, maxtoken, minbsize, pad_batch=pad_batch, batch_loader=batch_loader, pad_id=pad_id, **kwargs):

	for i_d, mlen_i in batch_loader(finput, bsize, maxpad, maxpart, maxtoken, minbsize, **kwargs):
		yield pad_batch(i_d, mlen_i, pad_id=pad_id)
