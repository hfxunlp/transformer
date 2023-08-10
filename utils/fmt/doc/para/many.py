#encoding: utf-8

from math import ceil

from utils.fmt.base import get_bsize, pad_batch
from utils.fmt.doc.base import doc_reader as file_reader
from utils.fmt.doc.para.single import batch_padder as batch_padder_single
from utils.fmt.vocab.base import map_batch

from cnfg.vocab.base import pad_id

def batch_loader_many(filelist, bsize, maxpad, maxpart, maxtoken, minbsize, get_bsize=get_bsize, file_reader=file_reader, **kwargs):

	_f_maxpart = float(maxpart)
	rs = [[] for i in range(len(filelist))]
	nd = maxlen = minlen = nsent = 0
	mlen = None
	for linelens in zip(*[file_reader(f) for f in filelist]):
		lines, lens = zip(*linelens)
		cur_nsent = len(lines[0])
		lgth = sum(lens)
		if maxlen == 0:
			_maxpad = max(1, min(maxpad, ceil(lgth / _f_maxpart)) // 2)
			maxlen = lgth + _maxpad
			minlen = lgth - _maxpad
			_bsize = max(1, get_bsize(lgth + _maxpad * len(lens), maxtoken, bsize) // cur_nsent)
			mlen = lens
		if (cur_nsent == nsent) and ((nd < minbsize) or (lgth <= maxlen and lgth >= minlen and nd < _bsize)):
			for line, rsu in zip(lines, rs):
				rsu.append(line)
			for cur_len, (i, mlenu,) in zip(lens, enumerate(mlen)):
				if cur_len > mlenu:
					mlen[i] = cur_len
			nd += 1
		else:
			yield rs, mlen, nsent
			rs = [[line] for line in lines]
			mlen = lens
			nsent = cur_nsent
			_maxpad = max(1, min(maxpad, ceil(lgth / _f_maxpart)) // 2)
			maxlen = lgth + _maxpad
			minlen = lgth - _maxpad
			_bsize = max(1, get_bsize(lgth + _maxpad * len(lens), maxtoken, bsize) // cur_nsent)
			nd = 1
	if rs:
		yield rs, mlen, nsent

def batch_mapper_many(filelist, vocablist, bsize, maxpad, maxpart, maxtoken, minbsize, map_batch=map_batch, batch_loader=batch_loader_many, **kwargs):

	for _rs, _mlen, nsent in batch_loader(filelist, bsize, maxpad, maxpart, maxtoken, minbsize, **kwargs):
		rs = []
		mlen = []
		for rsu, mlenu, vocab in zip(_rs, _mlen, vocablist):
			_rs, extok = map_batch(rsu, vocab)
			rs.append(_rs)
			mlen.append(mlenu + extok)
		yield rs, mlen, nsent

def batch_padder_many(filelist, vocablist, bsize, maxpad, maxpart, maxtoken, minbsize, pad_batch=pad_batch, batch_mapper=batch_mapper_many, pad_id=pad_id, **kwargs):

	for rs, mlen, nsent in batch_mapper(filelist, vocablist, bsize, maxpad, maxpart, maxtoken, minbsize, **kwargs):
		yield *tuple(pad_batch(rsu, mlenu, pad_id=pad_id) for rsu, mlenu in zip(rs, mlen)), nsent

def batch_padder(filelist, vocablist, bsize, maxpad, maxpart, maxtoken, minbsize, **kwargs):

	if isinstance(filelist, (list, tuple,)):
		if len(filelist) > 1:
			return batch_padder_many(filelist, vocablist, bsize, maxpad, maxpart, maxtoken, minbsize, **kwargs)
		else:
			return batch_padder_single(filelist[0], vocablist[0], bsize, maxpad, maxpart, maxtoken, minbsize, **kwargs)
	else:
		return batch_padder_single(filelist, vocablist, bsize, maxpad, maxpart, maxtoken, minbsize, **kwargs)
