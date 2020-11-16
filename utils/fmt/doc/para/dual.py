#encoding: utf-8

from utils.fmt.base import get_bsize, map_batch, pad_batch
from utils.fmt.doc.base import doc_reader

def batch_loader(finput, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize):

	rsi = []
	rst = []
	nd = maxlen = minlen = mlen_i = mlen_t = nsent = 0
	_bsize = bsize
	for (i_d, i_lgth), (td, t_lgth) in zip(doc_reader(finput), doc_reader(ftarget)):
		cur_nsent = len(i_d)
		lgth = i_lgth + t_lgth
		if maxlen == 0:
			_maxpad = max(1, min(maxpad, lgth // maxpart + 1) // 2)
			maxlen = lgth + _maxpad
			minlen = lgth - _maxpad
			_bsize = max(1, get_bsize(maxlen, maxtoken, bsize) // cur_nsent)
		if nsent == 0:
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
			_maxpad = max(1, min(maxpad, lgth // maxpart + 1) // 2)
			maxlen = lgth + _maxpad
			minlen = lgth - _maxpad
			_bsize = max(1, get_bsize(maxlen, maxtoken, bsize) // cur_nsent)
			nd = 1
	if rsi:
		yield rsi, rst, mlen_i, mlen_t, nsent

def batch_mapper(finput, ftarget, vocabi, vocabt, bsize, maxpad, maxpart, maxtoken, minbsize):

	for i_d, td, mlen_i, mlen_t, nsent in batch_loader(finput, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize):
		rsi, extok_i = map_batch(i_d, vocabi)
		rst, extok_t = map_batch(td, vocabt)
		yield rsi, rst, mlen_i + extok_i, mlen_t + extok_t, nsent

def batch_padder(finput, ftarget, vocabi, vocabt, bsize, maxpad, maxpart, maxtoken, minbsize):

	for i_d, td, mlen_i, mlen_t, nsent in batch_mapper(finput, ftarget, vocabi, vocabt, bsize, maxpad, maxpart, maxtoken, minbsize):
		yield pad_batch(i_d, mlen_i), pad_batch(td, mlen_t), nsent
