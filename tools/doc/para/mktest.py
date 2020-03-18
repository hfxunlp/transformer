#encoding: utf-8

import sys

import numpy
import h5py

from utils.fmt.base import ldvocab, dict2pairs
from utils.fmt.doc.para.single import batch_padder

def handle(finput, fvocab_i, frs, minbsize=1, expand_for_mulgpu=True, bsize=128, maxpad=16, maxpart=4, maxtoken=2048, minfreq=False, vsize=False):
	vcbi, nwordi = ldvocab(fvocab_i, minfreq, vsize)
	if expand_for_mulgpu:
		_bsize = bsize * minbsize
		_maxtoken = maxtoken * minbsize
	else:
		_bsize = bsize
		_maxtoken = maxtoken
	rsf = h5py.File(frs, 'w')
	src_grp = rsf.create_group("src")
	curd = {}
	for i_d, nsent in batch_padder(finput, vcbi, _bsize, maxpad, maxpart, _maxtoken, minbsize):
		rid = numpy.array(i_d, dtype = numpy.int32)
		_nsentgid = str(nsent)
		_curd = curd.get(nsent, 0)
		if _curd == 0:
			src_grp.create_group(_nsentgid)
		src_grp[_nsentgid][str(_curd)] = rid
		curd[nsent] = _curd + 1
	sents, ndl = dict2pairs(curd)
	rsf["nsent"] = numpy.array(sents, dtype = numpy.int32)
	rsf["ndata"] = numpy.array(ndl, dtype = numpy.int32)
	rsf["nword"] = numpy.array([nwordi], dtype = numpy.int32)
	rsf.close()
	print("Number of batches: %d\nSource Vocabulary Size: %d" % (sum(ndl), nwordi))

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
