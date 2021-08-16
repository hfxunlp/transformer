#encoding: utf-8

from utils.fmt.base import list_reader, get_bsize, map_batch, pad_batch

def batch_loader(finput, bsize, maxpad, maxpart, maxtoken, minbsize):

	rsi = []
	rstask = None
	nd = maxlen = minlen = mlen_i = 0
	_bsize = bsize
	for i_d in list_reader(finput):
		lgth = len(i_d) - 1
		_task = i_d[0]
		#if lgth <= 0:
			#continue
		if maxlen == 0:
			_maxpad = max(1, min(maxpad, lgth // maxpart + 1) // 2)
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
			_maxpad = max(1, min(maxpad, lgth // maxpart + 1) // 2)
			maxlen = lgth + _maxpad
			minlen = lgth - _maxpad
			_bsize = get_bsize(maxlen, maxtoken, bsize)
			nd = 1
	if rsi:
		yield rsi, rstask, mlen_i

def batch_mapper(finput, vocabi, vocabtask, bsize, maxpad, maxpart, maxtoken, minbsize):

	for i_d, taskd, mlen_i in batch_loader(finput, bsize, maxpad, maxpart, maxtoken, minbsize):
		rsi, extok_i = map_batch(i_d, vocabi)
		yield rsi, vocabtask[taskd], mlen_i + extok_i

def batch_padder(finput, vocabi, vocabtask, bsize, maxpad, maxpart, maxtoken, minbsize):

	for i_d, taskd, mlen_i in batch_mapper(finput, vocabi, vocabtask, bsize, maxpad, maxpart, maxtoken, minbsize):
		yield pad_batch(i_d, mlen_i), taskd
