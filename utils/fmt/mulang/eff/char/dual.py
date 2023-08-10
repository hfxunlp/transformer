#encoding: utf-8

from math import ceil

from utils.fmt.base import get_bsize, line_reader as file_reader
from utils.fmt.mulang.eff.dual import batch_padder as batch_padder_base

def batch_loader(finput, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize, get_bsize=get_bsize, file_reader=file_reader, **kwargs):

	_f_maxpart = float(maxpart)
	rsi = []
	rst = []
	rstask = None
	nd = maxlen = mlen_i = mlen_t = 0
	for i_d, td in zip(file_reader(finput, keep_empty_line=True), file_reader(ftarget, keep_empty_line=True)):
		_ind = i_d.find(" ")
		lid = len(i_d) - _ind - 1
		ltd = len(td)
		lgth = lid + ltd
		_task = i_d[:_ind]
		# uncomment the following 2 lines to filter out empty data (e.g. in OPUS-100).
		if (lid <= 0) or (ltd <= 0):
			continue
		if maxlen == 0:
			_maxpad = min(maxpad, ceil(lgth / _f_maxpart))
			maxlen = lgth + _maxpad
			_bsize = get_bsize(maxlen + _maxpad, maxtoken, bsize)
			rstask = _task
		if (rstask == _task) and ((nd < minbsize) or (lgth <= maxlen and nd < _bsize)):
			rsi.append(list(i_d[_ind + 1:]))
			rst.append(list(td))
			if lid > mlen_i:
				mlen_i = lid
			if ltd > mlen_t:
				mlen_t = ltd
			nd += 1
		else:
			yield rsi, rst, rstask, mlen_i, mlen_t
			rsi = [list(i_d[_ind + 1:])]
			rstask = _task
			rst = [list(td)]
			mlen_i = lid
			mlen_t = ltd
			_maxpad = min(maxpad, ceil(lgth / _f_maxpart))
			maxlen = lgth + _maxpad
			_bsize = get_bsize(maxlen + _maxpad, maxtoken, bsize)
			nd = 1
	if rsi:
		yield rsi, rst, rstask, mlen_i, mlen_t

def batch_padder(finput, ftarget, vocabi, vocabt, vocabtask, bsize, maxpad, maxpart, maxtoken, minbsize, batch_loader=batch_loader, **kwargs):

	return batch_padder_base(finput, ftarget, vocabi, vocabt, vocabtask, bsize, maxpad, maxpart, maxtoken, minbsize, batch_loader=batch_loader, **kwargs)
