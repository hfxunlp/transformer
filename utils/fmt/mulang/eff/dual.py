#encoding: utf-8

from utils.fmt.base import list_reader, get_bsize, map_batch, pad_batch

def batch_loader(finput, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize):

	rsi = []
	rst = []
	rstask = None
	nd = maxlen = mlen_i = mlen_t = 0
	for i_d, td in zip(list_reader(finput), list_reader(ftarget)):
		lid = len(i_d) - 1
		ltd = len(td)
		lgth = lid + ltd
		_task = i_d[0]
		# uncomment the following 2 lines to filter out empty data (e.g. in OPUS-100).
		#if (lid <= 0) or (ltd <= 0):
			#continue
		if maxlen == 0:
			maxlen = lgth + min(maxpad, lgth // maxpart + 1)
			_bsize = get_bsize(maxlen, maxtoken, bsize)
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
			maxlen = lgth + min(maxpad, lgth // maxpart + 1)
			_bsize = get_bsize(maxlen, maxtoken, bsize)
			nd = 1
	if rsi:
		yield rsi, rst, rstask, mlen_i, mlen_t

def batch_mapper(finput, ftarget, vocabi, vocabt, vocabtask, bsize, maxpad, maxpart, maxtoken, minbsize):

	for i_d, td, taskd, mlen_i, mlen_t in batch_loader(finput, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize):
		rsi, extok_i = map_batch(i_d, vocabi)
		rst, extok_t = map_batch(td, vocabt)
		yield rsi, rst, vocabtask[taskd], mlen_i + extok_i, mlen_t + extok_t

def batch_padder(finput, ftarget, vocabi, vocabt, vocabtask, bsize, maxpad, maxpart, maxtoken, minbsize):

	for i_d, td, taskd, mlen_i, mlen_t in batch_mapper(finput, ftarget, vocabi, vocabt, vocabtask, bsize, maxpad, maxpart, maxtoken, minbsize):
		yield pad_batch(i_d, mlen_i), pad_batch(td, mlen_t), taskd
