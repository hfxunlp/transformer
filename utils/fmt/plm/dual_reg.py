#encoding: utf-8

from math import ceil

from utils.fmt.base import get_bsize, iter_to_int, line_reader, list_reader, pad_batch

from cnfg.vocab.base import pad_id

file_reader = (list_reader, line_reader,)

def batch_loader(finput, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize, get_bsize=get_bsize, file_reader=file_reader, iter_to_int=iter_to_int, **kwargs):

	_f_maxpart = float(maxpart)
	rsi = []
	rst = []
	nd = maxlen = mlen_i = 0
	_list_reader, _line_reader = file_reader
	for i_d, td in zip(_list_reader(finput, keep_empty_line=True), _line_reader(ftarget, keep_empty_line=True)):
		i_d, td = list(iter_to_int(i_d)), float(td)
		lgth = len(i_d)
		if maxlen == 0:
			_maxpad = min(maxpad, ceil(lgth / _f_maxpart))
			maxlen = lgth + _maxpad
			_bsize = get_bsize(maxlen, maxtoken, bsize)
		if (nd < minbsize) or (lgth <= maxlen and nd < _bsize):
			rsi.append(i_d)
			rst.append(td)
			if lgth > mlen_i:
				mlen_i = lgth
			nd += 1
		else:
			yield rsi, rst, mlen_i
			rsi = [i_d]
			rst = [td]
			mlen_i = lgth
			_maxpad = min(maxpad, ceil(lgth / _f_maxpart))
			maxlen = lgth + _maxpad
			_bsize = get_bsize(maxlen, maxtoken, bsize)
			nd = 1
	if rsi:
		yield rsi, rst, mlen_i

def batch_padder(finput, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize, pad_batch=pad_batch, batch_loader=batch_loader, pad_id=pad_id, **kwargs):

	for i_d, td, mlen_i in batch_loader(finput, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize, **kwargs):
		yield pad_batch(i_d, mlen_i, pad_id=pad_id), td
