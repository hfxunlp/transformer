#encoding: utf-8

from utils.fmt.base import list_reader, has_unk, get_bsize, no_unk_mapper

def batch_loader(finput, fref, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize):

	rsi = []
	rsr = []
	rst = []
	nd = 0
	maxlen = 0
	_bsize = bsize
	mlen_i = 0
	mlen_r = 0
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

	global has_unk

	for i_d, rd, td, mlen_i, mlen_t in batch_loader(finput, fref, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize):
		rsi = []
		for lined in i_d:
			tmp = [1]
			tmp.extend([vocabi.get(wd, 3) for wd in lined] if has_unk else no_unk_mapper(vocabi, lined))#[vocabi[wd] for wd in lined if wd in vocabi]
			tmp.append(2)
			rsi.append(tmp)
		rsr = []
		for lined in rd:
			tmp = [1]
			tmp.extend([vocabi.get(wd, 3) for wd in lined] if has_unk else no_unk_mapper(vocabi, lined))#[vocabi[wd] for wd in lined if wd in vocabi]
			tmp.append(2)
			rsr.append(tmp)
		yield rsi, rsr, td, mlen_i + 2, mlen_t + 2

def batch_padder(finput, fref, ftarget, vocabi, bsize, maxpad, maxpart, maxtoken, minbsize):
	for i_d, rd, td, mlen_i, mlen_t in batch_mapper(finput, fref, ftarget, vocabi, bsize, maxpad, maxpart, maxtoken, minbsize):
		rid = []
		for lined in i_d:
			curlen = len(lined)
			if curlen < mlen_i:
				lined.extend([0 for i in range(mlen_i - curlen)])
			rid.append(lined)
		rrd = []
		for lined in rd:
			curlen = len(lined)
			if curlen < mlen_t:
				lined.extend([0 for i in range(mlen_t - curlen)])
			rrd.append(lined)
		rid.reverse()
		rrd.reverse()
		td.reverse()
		yield rid, rrd, td
