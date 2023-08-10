#encoding: utf-8

from math import ceil

from utils.fmt.base import get_bsize, pad_batch
from utils.fmt.doc.base import doc_reader as file_reader
from utils.fmt.vocab.base import map_batch

from cnfg.vocab.base import pad_id

def batch_loader(finput, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize, get_bsize=get_bsize, file_reader=file_reader, **kwargs):

	_f_maxpart = float(maxpart)
	rsi = []
	rst = []
	nd = maxlen = minlen = mlen_i = mlen_t = nsent = 0
	for (i_d, i_lgth), (td, t_lgth) in zip(file_reader(finput), file_reader(ftarget)):
		cur_nsent = len(i_d)
		lgth = i_lgth + t_lgth
		if maxlen == 0:
			_maxpad = max(1, min(maxpad, ceil(lgth / _f_maxpart)) // 2)
			maxlen = lgth + _maxpad
			minlen = lgth - _maxpad
			_bsize = max(1, get_bsize(maxlen + _maxpad, maxtoken, bsize) // cur_nsent)
			nsent = cur_nsent
		if (cur_nsent == nsent) and ((nd < minbsize) or (lgth <= maxlen and lgth >= minlen and nd < _bsize)):
			rsi.append(i_d)
			rst.append(td)
			if i_lgth > mlen_i:
				mlen_i = i_lgth
			if t_lgth > mlen_t:
				mlen_t = t_lgth
			nd += 1
		else:
			yield rsi, rst, mlen_i, mlen_t, nsent
			rsi = [i_d]
			rst = [td]
			mlen_i = i_lgth
			mlen_t = t_lgth
			nsent = cur_nsent
			_maxpad = max(1, min(maxpad, ceil(lgth / _f_maxpart)) // 2)
			maxlen = lgth + _maxpad
			minlen = lgth - _maxpad
			_bsize = max(1, get_bsize(maxlen + _maxpad, maxtoken, bsize) // cur_nsent)
			nd = 1
	if rsi:
		yield rsi, rst, mlen_i, mlen_t, nsent

def batch_mapper(finput, ftarget, vocabi, vocabt, bsize, maxpad, maxpart, maxtoken, minbsize, map_batch=map_batch, batch_loader=batch_loader, **kwargs):

	for i_d, td, mlen_i, mlen_t, nsent in batch_loader(finput, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize, **kwargs):
		rsi, extok_i = map_batch(i_d, vocabi)
		rst, extok_t = map_batch(td, vocabt)
		yield rsi, rst, mlen_i + extok_i, mlen_t + extok_t, nsent

def batch_padder(finput, ftarget, vocabi, vocabt, bsize, maxpad, maxpart, maxtoken, minbsize, pad_batch=pad_batch, batch_mapper=batch_mapper, pad_id=pad_id, **kwargs):

	for i_d, td, mlen_i, mlen_t, nsent in batch_mapper(finput, ftarget, vocabi, vocabt, bsize, maxpad, maxpart, maxtoken, minbsize, **kwargs):
		yield pad_batch(i_d, mlen_i, pad_id=pad_id), pad_batch(td, mlen_t, pad_id=pad_id), nsent
