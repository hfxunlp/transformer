#encoding: utf-8

from math import ceil

from utils.fmt.base import get_bsize, line_reader, list_reader, pad_batch
from utils.fmt.vocab.base import map_batch

from cnfg.vocab.base import pad_id

file_reader = (list_reader, line_reader,)

def batch_loader(finput, fref, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize, get_bsize=get_bsize, file_reader=file_reader, **kwargs):

	_f_maxpart = float(maxpart)
	rsi = []
	rsr = []
	rst = []
	nd = maxlen = mlen_i = mlen_r = 0
	_list_reader, _line_reader = file_reader
	for i_d, rd, td in zip(_list_reader(finput, keep_empty_line=True), _list_reader(fref, keep_empty_line=True), _line_reader(ftarget, keep_empty_line=True)):
		lid = len(i_d)
		lrd = len(rd)
		lgth = lid + lrd
		if maxlen == 0:
			_maxpad = min(maxpad, ceil(lgth / _f_maxpart))
			maxlen = lgth + _maxpad
			_bsize = get_bsize(maxlen + _maxpad, maxtoken, bsize)
		if (nd < minbsize) or (lgth <= maxlen and nd < _bsize):
			rsi.append(i_d)
			rsr.append(rd)
			rst.append(float(td))
			if lid > mlen_i:
				mlen_i = lid
			if lrd > mlen_r:
				mlen_r = lrd
			nd += 1
		else:
			yield rsi, rsr, rst, mlen_i, mlen_r
			rsi = [i_d]
			rsr = [rd]
			rst = [float(td)]
			mlen_i = lid
			mlen_r = lrd
			_maxpad = min(maxpad, ceil(lgth / _f_maxpart))
			maxlen = lgth + _maxpad
			_bsize = get_bsize(maxlen + _maxpad, maxtoken, bsize)
			nd = 1
	if rsi:
		yield rsi, rsr, rst, mlen_i, mlen_r

def batch_mapper(finput, fref, ftarget, vocabi, bsize, maxpad, maxpart, maxtoken, minbsize, map_batch=map_batch, batch_loader=batch_loader, **kwargs):

	for i_d, rd, td, mlen_i, mlen_t in batch_loader(finput, fref, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize, **kwargs):
		rsi, extok_i = map_batch(i_d, vocabi)
		rsr, extok_r = map_batch(rd, vocabi)
		yield rsi, rsr, td, mlen_i + extok_i, mlen_t + extok_r

def batch_padder(finput, fref, ftarget, vocabi, bsize, maxpad, maxpart, maxtoken, minbsize, pad_batch=pad_batch, batch_mapper=batch_mapper, pad_id=pad_id, **kwargs):

	for i_d, rd, td, mlen_i, mlen_t in batch_mapper(finput, fref, ftarget, vocabi, bsize, maxpad, maxpart, maxtoken, minbsize, **kwargs):
		yield pad_batch(i_d, mlen_i, pad_id=pad_id), pad_batch(rd, mlen_t, pad_id=pad_id), td
