#encoding: utf-8

from utils.fmt.base import list_reader, get_bsize, pad_batch, toint, pad_id
from math import ceil

def batch_loader(finput, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize):

	_f_maxpart = float(maxpart)
	rsi = []
	rst = []
	nd = maxlen = mlen_i = mlen_t = 0
	for i_d, td in zip(list_reader(finput, keep_empty_line=True), list_reader(ftarget, keep_empty_line=True)):
		i_d, td = toint(i_d), toint(td)
		lid = len(i_d)
		ltd = len(td)
		lgth = lid + ltd
		if maxlen == 0:
			maxlen = lgth + min(maxpad, ceil(lgth / _f_maxpart))
			_bsize = get_bsize(maxlen, maxtoken, bsize)
		if (nd < minbsize) or (lgth <= maxlen and nd < _bsize):
			rsi.append(i_d)
			rst.append(td)
			if lid > mlen_i:
				mlen_i = lid
			if ltd > mlen_t:
				mlen_t = ltd
			nd += 1
		else:
			yield rsi, rst, mlen_i, mlen_t
			rsi = [i_d]
			rst = [td]
			mlen_i = lid
			mlen_t = ltd
			maxlen = lgth + min(maxpad, ceil(lgth / _f_maxpart))
			_bsize = get_bsize(maxlen, maxtoken, bsize)
			nd = 1
	if rsi:
		yield rsi, rst, mlen_i, mlen_t

def batch_padder(finput, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=None, pad_id=pad_id, **kwargs):

	_batch_loader = batch_loader if custom_batch_loader is None else custom_batch_loader
	for i_d, td, mlen_i, mlen_t in _batch_loader(finput, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize):
		yield pad_batch(i_d, mlen_i, pad_id=pad_id), pad_batch(td, mlen_t, pad_id=pad_id)
