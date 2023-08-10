#encoding: utf-8

from math import ceil

from utils.fmt.base import get_bsize, list_reader as file_reader, pad_batch
from utils.fmt.vocab.base import map_batch

from cnfg.vocab.base import pad_id

def batch_loader(finput, bsize, maxpad, maxpart, maxtoken, minbsize, get_bsize=get_bsize, file_reader=file_reader, **kwargs):

	_f_maxpart = float(maxpart)
	rsi = []
	rstask = None
	nd = maxlen = minlen = mlen_i = 0
	for i_d in file_reader(finput, keep_empty_line=True):
		lgth = len(i_d) - 1
		_task = i_d[0]
		#if lgth <= 0:
			#continue
		if maxlen == 0:
			_maxpad = max(1, min(maxpad, ceil(lgth / _f_maxpart)) // 2)
			maxlen = lgth + _maxpad
			minlen = lgth - _maxpad
			_bsize = get_bsize(maxlen, maxtoken, bsize)
			rstask = _task
		if (rstask == _task) and ((nd < minbsize) or (lgth <= maxlen and lgth >= minlen and nd < _bsize)):
			rsi.append(i_d[1:])
			if lgth > mlen_i:
				mlen_i = lgth
			nd += 1
		else:
			yield rsi, rstask, mlen_i
			rsi = [i_d[1:]]
			rstask = _task
			mlen_i = lgth
			_maxpad = max(1, min(maxpad, ceil(lgth / _f_maxpart)) // 2)
			maxlen = lgth + _maxpad
			minlen = lgth - _maxpad
			_bsize = get_bsize(maxlen, maxtoken, bsize)
			nd = 1
	if rsi:
		yield rsi, rstask, mlen_i

def batch_mapper(finput, vocabi, vocabtask, bsize, maxpad, maxpart, maxtoken, minbsize, map_batch=map_batch, batch_loader=batch_loader, **kwargs):

	for i_d, taskd, mlen_i in batch_loader(finput, bsize, maxpad, maxpart, maxtoken, minbsize, **kwargs):
		rsi, extok_i = map_batch(i_d, vocabi)
		yield rsi, vocabtask[taskd], mlen_i + extok_i

def batch_padder(finput, vocabi, vocabtask, bsize, maxpad, maxpart, maxtoken, minbsize, pad_batch=pad_batch, batch_mapper=batch_mapper, pad_id=pad_id, **kwargs):

	for i_d, taskd, mlen_i in batch_mapper(finput, vocabi, vocabtask, bsize, maxpad, maxpart, maxtoken, minbsize, **kwargs):
		yield pad_batch(i_d, mlen_i, pad_id=pad_id), taskd
