#encoding: utf-8

import sys

import numpy
import h5py

has_unk = True

def list_reader(fname):
	def clear_list(lin):
		rs = []
		for tmpu in lin:
			if tmpu:
				rs.append(tmpu)
		return rs
	with open(fname, "rb") as frd:
		for line in frd:
			tmp = line.strip()
			if tmp:
				tmp = clear_list(tmp.decode("utf-8").split())
				yield tmp

def line_reader(fname):
	with open(fname, "rb") as frd:
		for line in frd:
			tmp = line.strip()
			if tmp:
				yield tmp.decode("utf-8")

def ldvocab(vfile, minf = False, omit_vsize = False):
	global has_unk
	if has_unk:
		rs = {"<pad>":0, "<sos>":1, "<eos>":2, "<unk>":3}
		cwd = 4
	else:
		rs = {"<pad>":0, "<sos>":1, "<eos>":2}
		cwd = 3
	if omit_vsize:
		vsize = omit_vsize
	else:
		vsize = False
	for data in list_reader(vfile):
		freq = int(data[0])
		if (not minf) or freq > minf:
			if vsize:
				ndata = len(data) - 1
				if vsize >= ndata:
					for wd in data[1:]:
						rs[wd] = cwd
						cwd += 1
				else:
					for wd in data[1:vsize + 1]:
						rs[wd] = cwd
						cwd += 1
						ndata = vsize
					break
				vsize -= ndata
				if vsize <= 0:
					break
			else:
				for wd in data[1:]:
					rs[wd] = cwd
					cwd += 1
		else:
			break
	return rs, cwd

def batch_loader(finput, bsize, maxpad, maxpart, maxtoken, minbsize):
	def get_bsize(maxlen, maxtoken, maxbsize):
		rs = max(maxtoken // maxlen, 1)
		if (rs % 2 == 1) and (rs > 1):
			rs -= 1
		return min(rs, maxbsize)
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

	def no_unk_mapper(vcb, ltm):
		rs = []
		for wd in ltm:
			if wd in vcb:
				rs.append(vcb[wd])
			else:
				print("Error mapping: "+ wd)
		return rs

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
		#ld = []
		rid = []
		for lined in i_d:
			curlen = len(lined)
			#ld.append(curlen)
			if curlen < mlen_i:
				lined.extend([0 for i in range(mlen_i - curlen)])
			rid.append(lined)
		#rid.reverse()
		yield rid

# maxtoken should be the maxtoken in mkiodata.py / 2 / beam size roughly, similar for bsize
def handle(finput, fvocab_i, frs, minbsize=1, expand_for_mulgpu=True, bsize=192, maxpad=16, maxpart=4, maxtoken=2560, minfreq = False, vsize = False):
	vcbi, nwordi = ldvocab(fvocab_i, minfreq, vsize)
	if expand_for_mulgpu:
		_bsize = bsize * minbsize
		_maxtoken = maxtoken * minbsize
	else:
		_bsize = bsize
		_maxtoken = maxtoken
	rsf = h5py.File(frs,'w')
	curd = 0
	for i_d in batch_padder(finput, vcbi, _bsize, maxpad, maxpart, _maxtoken, minbsize):
		rid = numpy.array(i_d, dtype = numpy.int32)
		#rld = numpy.array(ld, dtype = numpy.int32)
		wid = str(curd)
		rsf["i" + wid] = rid
		#rsf["l" + wid] = rld
		curd += 1
	rsf["ndata"] = numpy.array([curd], dtype = numpy.int32)
	rsf["nwordi"] = numpy.array([nwordi], dtype = numpy.int32)
	rsf.close()
	print("Number of batches: %d\nSource Vocabulary Size: %d" % (curd, nwordi))

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
