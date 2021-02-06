#encoding: utf-8

import sys

import numpy
import h5py

from utils.fmt.base import ldvocab, dict2pairs
from utils.fmt.doc.para.dual import batch_padder

from cnfg.ihyp import *

def handle(finput, ftarget, fvocab_i, fvocab_t, frs, minbsize=1, expand_for_mulgpu=True, bsize=max_sentences_gpu, maxpad=max_pad_tokens_sentence, maxpart=normal_tokens_vs_pad_tokens, maxtoken=max_tokens_gpu, minfreq=False, vsize=False):
	vcbi, nwordi = ldvocab(fvocab_i, minf=minfreq, omit_vsize=vsize, vanilla=False)
	vcbt, nwordt = ldvocab(fvocab_t, minf=minfreq, omit_vsize=vsize, vanilla=False)
	if expand_for_mulgpu:
		_bsize = bsize * minbsize
		_maxtoken = maxtoken * minbsize
	else:
		_bsize = bsize
		_maxtoken = maxtoken
	rsf = h5py.File(frs, 'w')
	src_grp = rsf.create_group("src")
	tgt_grp = rsf.create_group("tgt")
	curd = {}
	for i_d, td, nsent in batch_padder(finput, ftarget, vcbi, vcbt, _bsize, maxpad, maxpart, _maxtoken, minbsize):
		rid = numpy.array(i_d, dtype=numpy.int32)
		rtd = numpy.array(td, dtype=numpy.int32)
		_nsentgid = str(nsent)
		_curd = curd.get(nsent, 0)
		if _curd == 0:
			src_grp.create_group(_nsentgid)
			tgt_grp.create_group(_nsentgid)
		_curid = str(_curd)
		src_grp[_nsentgid].create_dataset(_curid, data=rid, **h5datawargs)
		tgt_grp[_nsentgid].create_dataset(_curid, data=rtd, **h5datawargs)
		curd[nsent] = _curd + 1
	sents, ndl = dict2pairs(curd)
	rsf["nsent"] = numpy.array(sents, dtype=numpy.int32)
	rsf["ndata"] = numpy.array(ndl, dtype=numpy.int32)
	rsf["nword"] = numpy.array([nwordi, nwordt], dtype=numpy.int32)
	rsf.close()
	print("Number of batches: %d\nSource Vocabulary Size: %d\nTarget Vocabulary Size: %d" % (sum(ndl), nwordi, nwordt,))

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]))
