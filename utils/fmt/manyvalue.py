#encoding: utf-8

from math import ceil

from utils.fmt.base import get_bsize, line_reader, list_reader, pad_batch
from utils.fmt.vocab.base import map_batch

from cnfg.vocab.base import pad_id

file_reader = (list_reader, line_reader,)

def batch_loader_many(filelist, bsize, maxpad, maxpart, maxtoken, minbsize, get_bsize=get_bsize, file_reader=file_reader, **kwargs):

	_f_maxpart = float(maxpart)
	rs = [[] for i in range(len(filelist))]
	nd = maxlen = 0
	mlen = None
	_list_reader, _line_reader = file_reader
	for lines in zip(*([_list_reader(f, keep_empty_line=True) for f in filelist[:-1]] + [_line_reader(filelist[-1], keep_empty_line=True)])):
		lens = [len(line) for line in lines[:-1]]
		lgth = sum(lens)
		if maxlen == 0:
			_maxpad = min(maxpad, ceil(lgth / _f_maxpart))
			maxlen = lgth + _maxpad
			_bsize = get_bsize(lgth + _maxpad * len(lens), maxtoken, bsize)
			mlen = lens
		if (nd < minbsize) or (lgth <= maxlen and nd < _bsize):
			for line, rsu in zip(lines[:-1], rs):
				rsu.append(line)
			rs[-1].append(float(lines[-1]))
			for cur_len, (i, mlenu,) in zip(lens, enumerate(mlen)):
				if cur_len > mlenu:
					mlen[i] = cur_len
			nd += 1
		else:
			yield rs, mlen
			rs = [[line] for line in lines[:-1]]
			rs.append([float(lines[-1])])
			mlen = lens
			_maxpad = min(maxpad, ceil(lgth / _f_maxpart))
			maxlen = lgth + _maxpad
			_bsize = get_bsize(lgth + _maxpad * len(lens), maxtoken, bsize)
			nd = 1
	if rs:
		yield rs, mlen

def batch_mapper_many(filelist, vocablist, bsize, maxpad, maxpart, maxtoken, minbsize, map_batch=map_batch, batch_loader=batch_loader_many, **kwargs):

	for _rs, _mlen in batch_loader(filelist, bsize, maxpad, maxpart, maxtoken, minbsize, **kwargs):
		rs = []
		mlen = []
		for rsu, mlenu, vocab in zip(_rs, _mlen, vocablist):
			_rs, extok = map_batch(rsu, vocab)
			rs.append(_rs)
			mlen.append(mlenu + extok)
		rs.append(_rs[-1])
		yield rs, mlen

def batch_padder(filelist, vocablist, bsize, maxpad, maxpart, maxtoken, minbsize, pad_batch=pad_batch, batch_mapper=batch_mapper_many, pad_id=pad_id, **kwargs):

	for rs, mlen in batch_mapper(filelist, vocablist, bsize, maxpad, maxpart, maxtoken, minbsize, **kwargs):
		yield *tuple(pad_batch(rsu, mlenu, pad_id=pad_id) for rsu, mlenu in zip(rs, mlen)), rs[-1]
