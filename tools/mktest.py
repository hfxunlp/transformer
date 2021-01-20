#encoding: utf-8

import sys

import numpy
import h5py

from utils.fmt.base import ldvocab
from utils.fmt.single import batch_padder

from cnfg.ihyp import *

# maxtoken should be the maxtoken in mkiodata.py / 2 / beam size roughly, similar for bsize

def handle(finput, fvocab_i, frs, minbsize=1, expand_for_mulgpu=True, bsize=max_sentences_gpu, maxpad=max_pad_tokens_sentence, maxpart=normal_tokens_vs_pad_tokens, maxtoken=max_tokens_gpu, minfreq=False, vsize=False):
	vcbi, nwordi = ldvocab(fvocab_i, minf=minfreq, omit_vsize=vsize, vanilla=False)
	if expand_for_mulgpu:
		_bsize = bsize * minbsize
		_maxtoken = maxtoken * minbsize
	else:
		_bsize = bsize
		_maxtoken = maxtoken
	rsf = h5py.File(frs,'w')
	src_grp = rsf.create_group("src")
	curd = 0
	for i_d in batch_padder(finput, vcbi, _bsize, maxpad, maxpart, _maxtoken, minbsize):
		rid = numpy.array(i_d, dtype=numpy.int32)
		#rld = numpy.array(ld, dtype=numpy.int32)
		wid = str(curd)
		src_grp.create_dataset(wid, data=rid, **h5datawargs)
		#rsf["l" + wid] = rld
		curd += 1
	rsf["ndata"] = numpy.array([curd], dtype=numpy.int32)
	rsf["nword"] = numpy.array([nwordi], dtype=numpy.int32)
	rsf.close()
	print("Number of batches: %d\nSource Vocabulary Size: %d" % (curd, nwordi,))

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
