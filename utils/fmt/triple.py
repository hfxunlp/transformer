#encoding: utf-8

from utils.fmt.base import list_reader, get_bsize, map_batch, pad_batch

def batch_loader(finput, fref, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize):

	rsi = []
	rsr = []
	rst = []
	nd = maxlen = mlen_i = mlen_r = 0
	_bsize = bsize
	for i_d, rd, td in zip(list_reader(finput), list_reader(fref), line_reader(ftarget)):
		lid = len(i_d)
		lrd = len(rd)
		lgth = lid + lrd
		if maxlen == 0:
			maxlen = lgth + min(maxpad, lgth // maxpart + 1)
			_bsize = get_bsize(maxlen, maxtoken, bsize)
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
			maxlen = lgth + min(maxpad, lgth // maxpart + 1)
			_bsize = get_bsize(maxlen, maxtoken, bsize)
			nd = 1
	if rsi:
		yield rsi, rsr, rst, mlen_i, mlen_r

def batch_mapper(finput, fref, ftarget, vocabi, bsize, maxpad, maxpart, maxtoken, minbsize):

	for i_d, rd, td, mlen_i, mlen_t in batch_loader(finput, fref, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize):
		rsi, extok_i = map_batch(i_d, vocabi)
		rsr, extok_r = map_batch(rd, vocabi)
		yield rsi, rsr, td, mlen_i + extok_i, mlen_t + extok_r

def batch_padder(finput, fref, ftarget, vocabi, bsize, maxpad, maxpart, maxtoken, minbsize):

	for i_d, rd, td, mlen_i, mlen_t in batch_mapper(finput, fref, ftarget, vocabi, bsize, maxpad, maxpart, maxtoken, minbsize):
		yield pad_batch(i_d, mlen_i), pad_batch(rd, mlen_t), td
