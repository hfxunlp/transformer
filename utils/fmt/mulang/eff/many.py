#encoding: utf-8

from math import ceil

from utils.fmt.base import get_bsize, list_reader as file_reader, pad_batch
from utils.fmt.mulang.eff.single import batch_padder as batch_padder_single
from utils.fmt.vocab.base import map_batch

from cnfg.vocab.base import pad_id

def batch_loader_many(filelist, bsize, maxpad, maxpart, maxtoken, minbsize, get_bsize=get_bsize, file_reader=file_reader, **kwargs):

	_f_maxpart = float(maxpart)
	rs = [[] for i in range(len(filelist))]
	nd = maxlen = 0
	mlen = rstask = None
	for lines in zip(*[file_reader(f, keep_empty_line=True) for f in filelist]):
		lens = [len(line) for line in lines]
		lens[0] -= 1
		lgth = sum(lens)
		src_line = lines[0]
		_task = src_line[0]
		# uncomment the following 2 lines to filter out empty data (e.g. in OPUS-100).
		if any(_len <= 0 for _len in lens):
			continue
		if maxlen == 0:
			_maxpad = min(maxpad, ceil(lgth / _f_maxpart))
			maxlen = lgth + _maxpad
			_bsize = get_bsize(lgth + _maxpad * len(lens), maxtoken, bsize)
			mlen = lens
			rstask = _task
		if (rstask == _task) and ((nd < minbsize) or (lgth <= maxlen and nd < _bsize)):
			rs[0].append(src_line[1:])
			for line, rsu in zip(lines[1:], rs[1:]):
				rsu.append(line)
			for cur_len, (i, mlenu,) in zip(lens, enumerate(mlen)):
				if cur_len > mlenu:
					mlen[i] = cur_len
			nd += 1
		else:
			yield rs, rstask, mlen
			rs = [[src_line[1:]]]
			rs.extend([[line] for line in lines[1:]])
			mlen = lens
			rstask = _task
			_maxpad = min(maxpad, ceil(lgth / _f_maxpart))
			maxlen = lgth + _maxpad
			_bsize = get_bsize(lgth + _maxpad * len(lens), maxtoken, bsize)
			nd = 1
	if rs:
		yield rs, rstask, mlen

def batch_mapper_many(filelist, vocablist, bsize, maxpad, maxpart, maxtoken, minbsize, map_batch=map_batch, batch_loader=batch_loader_many, **kwargs):

	vocabtask = vocablist[-1]
	for _rs, taskd, _mlen in batch_loader(filelist, bsize, maxpad, maxpart, maxtoken, minbsize, **kwargs):
		rs = []
		mlen = []
		for rsu, mlenu, vocab in zip(_rs, _mlen, vocablist):
			_rs, extok = map_batch(rsu, vocab)
			rs.append(_rs)
			mlen.append(mlenu + extok)
		yield rs, vocabtask[taskd], mlen

def batch_padder_many(filelist, vocablist, bsize, maxpad, maxpart, maxtoken, minbsize, pad_batch=pad_batch, batch_mapper=batch_mapper_many, pad_id=pad_id, **kwargs):

	for rs, taskd, mlen in batch_mapper(filelist, vocablist, bsize, maxpad, maxpart, maxtoken, minbsize, **kwargs):
		yield *tuple(pad_batch(rsu, mlenu, pad_id=pad_id) for rsu, mlenu in zip(rs, mlen)), taskd

def batch_padder(filelist, vocablist, bsize, maxpad, maxpart, maxtoken, minbsize, **kwargs):

	if isinstance(filelist, (list, tuple,)):
		if len(filelist) > 1:
			return batch_padder_many(filelist, vocablist, bsize, maxpad, maxpart, maxtoken, minbsize, **kwargs)
		else:
			return batch_padder_single(filelist[0], *vocablist, bsize, maxpad, maxpart, maxtoken, minbsize, **kwargs)
	else:
		return batch_padder_single(filelist, *vocablist, bsize, maxpad, maxpart, maxtoken, minbsize, **kwargs)
