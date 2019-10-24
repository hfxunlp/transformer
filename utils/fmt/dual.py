#encoding: utf-8

from utils.fmt.base import list_reader, has_unk, get_bsize, no_unk_mapper

def batch_loader(finput, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize):

	rsi = []
	rst = []
	nd = 0
	maxlen = 0
	_bsize = bsize
	mlen_i = 0
	mlen_t = 0
	for i_d, td in zip(list_reader(finput), list_reader(ftarget)):
		lid = len(i_d)
		ltd = len(td)
		lgth = lid + ltd
		if maxlen == 0:
			maxlen = lgth + min(maxpad, lgth // maxpart + 1)
			_bsize = get_bsize(maxlen, maxtoken, bsize)
		if (nd < minbsize) or (lgth <= maxlen and nd < _bsize):
			rsi.append(i_d)
			rst.append(td)
			if lid > mlen_i:
				mlen_i = lid
			if ltd > mlen_t:
				mlen_t = ltd
			nd += 1
		else:
			yield rsi, rst, mlen_i, mlen_t
			rsi = [i_d]
			rst = [td]
			mlen_i = lid
			mlen_t = ltd
			maxlen = lgth + min(maxpad, lgth // maxpart + 1)
			_bsize = get_bsize(maxlen, maxtoken, bsize)
			nd = 1
	if rsi:
		yield rsi, rst, mlen_i, mlen_t

def batch_mapper(finput, ftarget, vocabi, vocabt, bsize, maxpad, maxpart, maxtoken, minbsize):

	global has_unk

	for i_d, td, mlen_i, mlen_t in batch_loader(finput, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize):
		rsi = []
		for lined in i_d:
			tmp = [1]
			tmp.extend([vocabi.get(wd, 3) for wd in lined] if has_unk else no_unk_mapper(vocabi, lined))#[vocabi[wd] for wd in lined if wd in vocabi]
			tmp.append(2)
			rsi.append(tmp)
		rst = []
		for lined in td:
			tmp = [1]
			tmp.extend([vocabt.get(wd, 3) for wd in lined] if has_unk else no_unk_mapper(vocabt, lined))#[vocabt[wd] for wd in lined if wd in vocabt]
			tmp.append(2)
			rst.append(tmp)
		yield rsi, rst, mlen_i + 2, mlen_t + 2

def batch_padder(finput, ftarget, vocabi, vocabt, bsize, maxpad, maxpart, maxtoken, minbsize):
	for i_d, td, mlen_i, mlen_t in batch_mapper(finput, ftarget, vocabi, vocabt, bsize, maxpad, maxpart, maxtoken, minbsize):
		#ld = []
		rid = []
		for lined in i_d:
			curlen = len(lined)
			#ld.append(curlen)
			if curlen < mlen_i:
				lined.extend([0 for i in range(mlen_i - curlen)])
			rid.append(lined)
		rtd = []
		for lined in td:
			curlen = len(lined)
			if curlen < mlen_t:
				lined.extend([0 for i in range(mlen_t - curlen)])
			rtd.append(lined)
		#rid.reverse()
		#rtd.reverse()
		#ld.reverse()
		yield rid, rtd
