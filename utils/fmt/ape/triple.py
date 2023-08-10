#encoding: utf-8

from math import ceil

from utils.fmt.base import get_bsize, list_reader as file_reader, pad_batch
from utils.fmt.vocab.base import map_batch

from cnfg.vocab.base import pad_id

def batch_loader(finput, fmt, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize, get_bsize=get_bsize, file_reader=file_reader, **kwargs):

	_f_maxpart = float(maxpart)
	rsi = []
	rsm = []
	rst = []
	nd = maxlen = mlen_i = mlen_m = mlen_t = 0
	for i_d, md, td in zip(file_reader(finput, keep_empty_line=True), file_reader(fmt, keep_empty_line=True), file_reader(ftarget, keep_empty_line=True)):
		lid = len(i_d)
		lmd = len(md)
		ltd = len(td)
		lgth = lid + lmd + ltd
		if maxlen == 0:
			_maxpad = min(maxpad, ceil(lgth / _f_maxpart))
			maxlen = lgth + _maxpad
			_bsize = get_bsize(lgth + _maxpad * 3, maxtoken, bsize)
		if (nd < minbsize) or (lgth <= maxlen and nd < _bsize):
			rsi.append(i_d)
			rsm.append(md)
			rst.append(td)
			if lid > mlen_i:
				mlen_i = lid
			if lmd > mlen_m:
				mlen_m = lmd
			if ltd > mlen_t:
				mlen_t = ltd
			nd += 1
		else:
			yield rsi, rsm, rst, mlen_i, mlen_m, mlen_t
			rsi = [i_d]
			rsm = [md]
			rst = [td]
			mlen_i = lid
			mlen_m = lmd
			mlen_t = ltd
			_maxpad = min(maxpad, ceil(lgth / _f_maxpart))
			maxlen = lgth + _maxpad
			_bsize = get_bsize(lgth + _maxpad * 3, maxtoken, bsize)
			nd = 1
	if rsi:
		yield rsi, rsm, rst, mlen_i, mlen_m, mlen_t

def batch_mapper(finput, fmt, ftarget, vocabi, vocabt, bsize, maxpad, maxpart, maxtoken, minbsize, map_batch=map_batch, batch_loader=batch_loader, **kwargs):

	for i_d, md, td, mlen_i, mlen_m, mlen_t in batch_loader(finput, fmt, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize, **kwargs):
		rsi, extok_i = map_batch(i_d, vocabi)
		rsm, extok_m = map_batch(md, vocabt)
		rst, extok_t = map_batch(td, vocabt)
		yield rsi, rsm, rst, mlen_i + extok_i, mlen_m + extok_m, mlen_t + extok_t

def batch_padder(finput, fmt, ftarget, vocabi, vocabt, bsize, maxpad, maxpart, maxtoken, minbsize, pad_batch=pad_batch, batch_mapper=batch_mapper, pad_id=pad_id, **kwargs):

	for i_d, md, td, mlen_i, mlen_m, mlen_t in batch_mapper(finput, fmt, ftarget, vocabi, vocabt, bsize, maxpad, maxpart, maxtoken, minbsize, **kwargs):
		yield pad_batch(i_d, mlen_i, pad_id=pad_id), pad_batch(md, mlen_m, pad_id=pad_id), pad_batch(td, mlen_t, pad_id=pad_id)
