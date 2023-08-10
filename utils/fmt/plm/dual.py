#encoding: utf-8

from math import ceil

from utils.fmt.base import get_bsize, iter_to_int, list_reader as file_reader, pad_batch

from cnfg.vocab.base import pad_id

def batch_loader(finput, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize, get_bsize=get_bsize, file_reader=file_reader, iter_to_int=iter_to_int, **kwargs):

	_f_maxpart = float(maxpart)
	rsi = []
	rst = []
	nd = maxlen = mlen_i = mlen_t = 0
	for i_d, td in zip(file_reader(finput, keep_empty_line=True), file_reader(ftarget, keep_empty_line=True)):
		i_d, td = list(iter_to_int(i_d)), list(iter_to_int(td))
		lid = len(i_d)
		ltd = len(td)
		lgth = lid + ltd
		if maxlen == 0:
			_maxpad = min(maxpad, ceil(lgth / _f_maxpart))
			maxlen = lgth + _maxpad
			_bsize = get_bsize(maxlen + _maxpad, maxtoken, bsize)
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
			_maxpad = min(maxpad, ceil(lgth / _f_maxpart))
			maxlen = lgth + _maxpad
			_bsize = get_bsize(maxlen + _maxpad, maxtoken, bsize)
			nd = 1
	if rsi:
		yield rsi, rst, mlen_i, mlen_t

def batch_padder(finput, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize, pad_batch=pad_batch, batch_loader=batch_loader, pad_id=pad_id, **kwargs):

	for i_d, td, mlen_i, mlen_t in batch_loader(finput, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize, **kwargs):
		yield pad_batch(i_d, mlen_i, pad_id=pad_id), pad_batch(td, mlen_t, pad_id=pad_id)
