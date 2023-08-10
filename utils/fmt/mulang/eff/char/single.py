#encoding: utf-8

from math import ceil

from utils.fmt.base import get_bsize, line_reader as file_reader
from utils.fmt.mulang.eff.single import batch_padder as batch_padder_base

def batch_loader(finput, bsize, maxpad, maxpart, maxtoken, minbsize, get_bsize=get_bsize, file_reader=file_reader, **kwargs):

	_f_maxpart = float(maxpart)
	rsi = []
	rstask = None
	nd = maxlen = minlen = mlen_i = 0
	for i_d in file_reader(finput, keep_empty_line=True):
		_ind = i_d.find(" ")
		lgth = len(i_d) - _ind - 1
		_task = i_d[:_ind]
		#if lgth <= 0:
			#continue
		if maxlen == 0:
			_maxpad = max(1, min(maxpad, ceil(lgth / _f_maxpart)) // 2)
			maxlen = lgth + _maxpad
			minlen = lgth - _maxpad
			_bsize = get_bsize(maxlen, maxtoken, bsize)
			rstask = _task
		if (rstask == _task) and ((nd < minbsize) or (lgth <= maxlen and lgth >= minlen and nd < _bsize)):
			rsi.append(list(i_d[_ind + 1:]))
			if lgth > mlen_i:
				mlen_i = lgth
			nd += 1
		else:
			yield rsi, rstask, mlen_i
			rsi = [list(i_d[_ind + 1:])]
			rstask = _task
			mlen_i = lgth
			_maxpad = max(1, min(maxpad, ceil(lgth / _f_maxpart)) // 2)
			maxlen = lgth + _maxpad
			minlen = lgth - _maxpad
			_bsize = get_bsize(maxlen, maxtoken, bsize)
			nd = 1
	if rsi:
		yield rsi, rstask, mlen_i

def batch_padder(finput, vocabi, vocabtask, bsize, maxpad, maxpart, maxtoken, minbsize, batch_loader=batch_loader, **kwargs):

	return batch_padder_base(finput, vocabi, vocabtask, bsize, maxpad, maxpart, maxtoken, minbsize, batch_loader=batch_loader, **kwargs)
