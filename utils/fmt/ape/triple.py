#encoding: utf-8

from utils.fmt.base import list_reader, get_bsize, map_batch, pad_batch
from math import ceil

def batch_loader(finput, fmt, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize):

	_f_maxpart = float(maxpart)
	rsi = []
	rsm = []
	rst = []
	nd = maxlen = mlen_i = mlen_m = mlen_t = 0
	for i_d, md, td in zip(list_reader(finput, keep_empty_line=True), list_reader(fmt, keep_empty_line=True), list_reader(ftarget, keep_empty_line=True)):
		lid = len(i_d)
		lmd = len(md)
		ltd = len(td)
		lgth = lid + lmd + ltd
		if maxlen == 0:
			maxlen = lgth + min(maxpad, ceil(lgth / _f_maxpart))
			_bsize = get_bsize(maxlen, maxtoken, bsize)
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
			maxlen = lgth + min(maxpad, ceil(lgth / _f_maxpart))
			_bsize = get_bsize(maxlen, maxtoken, bsize)
			nd = 1
	if rsi:
		yield rsi, rsm, rst, mlen_i, mlen_m, mlen_t

def batch_mapper(finput, fmt, ftarget, vocabi, vocabt, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=None):

	_batch_loader = batch_loader if custom_batch_loader is None else custom_batch_loader
	for i_d, md, td, mlen_i, mlen_m, mlen_t in _batch_loader(finput, fmt, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize):
		rsi, extok_i = map_batch(i_d, vocabi)
		rsm, extok_m = map_batch(md, vocabt)
		rst, extok_t = map_batch(td, vocabt)
		yield rsi, rsm, rst, mlen_i + extok_i, mlen_m + extok_m, mlen_t + extok_t

def batch_padder(finput, fmt, ftarget, vocabi, vocabt, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=None, custom_batch_mapper=None):

	_batch_mapper = batch_mapper if custom_batch_mapper is None else custom_batch_mapper
	for i_d, md, td, mlen_i, mlen_m, mlen_t in _batch_mapper(finput, fmt, ftarget, vocabi, vocabt, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=custom_batch_loader):
		yield pad_batch(i_d, mlen_i), pad_batch(md, mlen_m), pad_batch(td, mlen_t)
