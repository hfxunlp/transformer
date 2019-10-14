#encoding: utf-8

from utils.fmt.base import list_reader, has_unk, get_bsize, no_unk_mapper

def batch_loader(finput, bsize, maxpad, maxpart, maxtoken, minbsize):

	rsi = []
	nd = 0
	maxlen = 0
	minlen = 0
	_bsize = bsize
	mlen_i = 0
	for i_d in list_reader(finput):
		lgth = len(i_d)
		if maxlen == 0:
			_maxpad = max(1, min(maxpad, lgth // maxpart + 1) // 2)
			maxlen = lgth + _maxpad
			minlen = lgth - _maxpad
			_bsize = get_bsize(maxlen, maxtoken, bsize)
		if (nd < minbsize) or (lgth <= maxlen and lgth >= minlen and nd < _bsize):
			rsi.append(i_d)
			if lgth > mlen_i:
				mlen_i = lgth
			nd += 1
		else:
			yield rsi, mlen_i
			rsi = [i_d]
			mlen_i = lgth
			_maxpad = max(1, min(maxpad, lgth // maxpart + 1) // 2)
			maxlen = lgth + _maxpad
			minlen = lgth - _maxpad
			_bsize = get_bsize(maxlen, maxtoken, bsize)
			nd = 1
	if rsi:
		yield rsi, mlen_i

def batch_mapper(finput, vocabi, bsize, maxpad, maxpart, maxtoken, minbsize):

	global has_unk

	for i_d, mlen_i in batch_loader(finput, bsize, maxpad, maxpart, maxtoken, minbsize):
		rsi = []
		for lined in i_d:
			tmp = [1]
			tmp.extend([vocabi.get(wd, 3) for wd in lined] if has_unk else no_unk_mapper(vocabi, lined))#[vocabi[wd] for wd in lined if wd in vocabi]
			tmp.append(2)
			rsi.append(tmp)
		yield rsi, mlen_i + 2

def batch_padder(finput, vocabi, bsize, maxpad, maxpart, maxtoken, minbsize):
	for i_d, mlen_i in batch_mapper(finput, vocabi, bsize, maxpad, maxpart, maxtoken, minbsize):
		rid = []
		for lined in i_d:
			curlen = len(lined)
			if curlen < mlen_i:
				lined.extend([0 for i in range(mlen_i - curlen)])
			rid.append(lined)
		yield rid
