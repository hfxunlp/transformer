#encoding: utf-8

from math import ceil

from utils.fmt.base import get_bsize, list_reader as file_reader, pad_batch
from utils.fmt.vocab.base import map_batch

from cnfg.vocab.base import pad_id

def batch_loader(finput, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize, get_bsize=get_bsize, file_reader=file_reader, **kwargs):

	_f_maxpart = float(maxpart)
	rsi = []
	rst = []
	rstask = None
	nd = maxlen = mlen_i = mlen_t = 0
	for i_d, td in zip(file_reader(finput, keep_empty_line=True), file_reader(ftarget, keep_empty_line=True)):
		lid = len(i_d) - 1
		ltd = len(td)
		lgth = lid + ltd
		_task = i_d[0]
		# uncomment the following 2 lines to filter out empty data (e.g. in OPUS-100).
		if (lid <= 0) or (ltd <= 0):
			continue
		if maxlen == 0:
			_maxpad = min(maxpad, ceil(lgth / _f_maxpart))
			maxlen = lgth + _maxpad
			_bsize = get_bsize(maxlen + _maxpad, maxtoken, bsize)
			rstask = _task
		if (rstask == _task) and ((nd < minbsize) or (lgth <= maxlen and nd < _bsize)):
			rsi.append(i_d[1:])
			rst.append(td)
			if lid > mlen_i:
				mlen_i = lid
			if ltd > mlen_t:
				mlen_t = ltd
			nd += 1
		else:
			yield rsi, rst, rstask, mlen_i, mlen_t
			rsi = [i_d[1:]]
			rstask = _task
			rst = [td]
			mlen_i = lid
			mlen_t = ltd
			_maxpad = min(maxpad, ceil(lgth / _f_maxpart))
			maxlen = lgth + _maxpad
			_bsize = get_bsize(maxlen + _maxpad, maxtoken, bsize)
			nd = 1
	if rsi:
		yield rsi, rst, rstask, mlen_i, mlen_t

def batch_mapper(finput, ftarget, vocabi, vocabt, vocabtask, bsize, maxpad, maxpart, maxtoken, minbsize, map_batch=map_batch, batch_loader=batch_loader, **kwargs):

	for i_d, td, taskd, mlen_i, mlen_t in batch_loader(finput, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize, **kwargs):
		rsi, extok_i = map_batch(i_d, vocabi)
		rst, extok_t = map_batch(td, vocabt)
		yield rsi, rst, vocabtask[taskd], mlen_i + extok_i, mlen_t + extok_t

def batch_padder(finput, ftarget, vocabi, vocabt, vocabtask, bsize, maxpad, maxpart, maxtoken, minbsize, pad_batch=pad_batch, batch_mapper=batch_mapper, pad_id=pad_id, **kwargs):

	for i_d, td, taskd, mlen_i, mlen_t in batch_mapper(finput, ftarget, vocabi, vocabt, vocabtask, bsize, maxpad, maxpart, maxtoken, minbsize, **kwargs):
		yield pad_batch(i_d, mlen_i, pad_id=pad_id), pad_batch(td, mlen_t, pad_id=pad_id), taskd
