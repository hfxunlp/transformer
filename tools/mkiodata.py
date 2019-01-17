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

def batch_loader(finput, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize):
	def get_bsize(maxlen, maxtoken, maxbsize):
		rs = max(maxtoken // maxlen, 1)
		if (rs % 2 == 1) and (rs > 1):
			rs -= 1
		return min(rs, maxbsize)
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

	def no_unk_mapper(vcb, ltm):
		rs = []
		for wd in ltm:
			if wd in vcb:
				rs.append(vcb[wd])
			else:
				print("Error mapping: "+ wd)
		return rs

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
		rid.reverse()
		rtd.reverse()
		#ld.reverse()
		yield rid, rtd

def handle(finput, ftarget, fvocab_i, fvocab_t, frs, minbsize=1, expand_for_mulgpu=True, bsize=768, maxpad=16, maxpart=4, maxtoken=3920, minfreq = False, vsize = False):
	vcbi, nwordi = ldvocab(fvocab_i, minfreq, vsize)
	vcbt, nwordt = ldvocab(fvocab_t, minfreq, vsize)
	if expand_for_mulgpu:
		_bsize = bsize * minbsize
		_maxtoken = maxtoken * minbsize
	else:
		_bsize = bsize
		_maxtoken = maxtoken
	rsf = h5py.File(frs, 'w')
	curd = 0
	for i_d, td in batch_padder(finput, ftarget, vcbi, vcbt, _bsize, maxpad, maxpart, _maxtoken, minbsize):
		rid = numpy.array(i_d, dtype = numpy.int32)
		rtd = numpy.array(td, dtype = numpy.int32)
		#rld = numpy.array(ld, dtype = numpy.int32)
		wid = str(curd)
		rsf["i" + wid] = rid
		rsf["t" + wid] = rtd
		#rsf["l" + wid] = rld
		curd += 1
	rsf["ndata"] = numpy.array([curd], dtype = numpy.int32)
	rsf["nwordi"] = numpy.array([nwordi], dtype = numpy.int32)
	rsf["nwordt"] = numpy.array([nwordt], dtype = numpy.int32)
	rsf.close()
	print("Number of batches: %d\nSource Vocabulary Size: %d\nTarget Vocabulary Size: %d" % (curd, nwordi, nwordt))

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]))
