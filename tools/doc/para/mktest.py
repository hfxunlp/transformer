#encoding: utf-8

import sys
from numpy import array as np_array, int32 as np_int32

from utils.fmt.base import dict2pairs
from utils.fmt.doc.para.single import batch_padder
from utils.fmt.vocab.token import ldvocab
from utils.h5serial import h5File

from cnfg.ihyp import *

def handle(finput, fvocab_i, frs, minbsize=1, expand_for_mulgpu=True, bsize=max_sentences_gpu, maxpad=max_pad_tokens_sentence, maxpart=normal_tokens_vs_pad_tokens, maxtoken=max_tokens_gpu, minfreq=False, vsize=False):
	vcbi, nwordi = ldvocab(fvocab_i, minf=minfreq, omit_vsize=vsize, vanilla=False)
	if expand_for_mulgpu:
		_bsize = bsize * minbsize
		_maxtoken = maxtoken * minbsize
	else:
		_bsize = bsize
		_maxtoken = maxtoken
	with h5File(frs, "w", libver=h5_libver) as rsf:
		src_grp = rsf.create_group("src")
		curd = {}
		for i_d, nsent in batch_padder(finput, vcbi, _bsize, maxpad, maxpart, _maxtoken, minbsize):
			rid = np_array(i_d, dtype=np_int32)
			_nsentgid = str(nsent)
			_curd = curd.get(nsent, 0)
			if _curd == 0:
				src_grp.create_group(_nsentgid)
			src_grp[_nsentgid].create_dataset(str(_curd), data=rid, **h5datawargs)
			curd[nsent] = _curd + 1
		sents, ndl = dict2pairs(curd)
		rsf["nsent"] = np_array(sents, dtype=np_int32)
		rsf["ndata"] = np_array(ndl, dtype=np_int32)
		rsf["nword"] = np_array([nwordi], dtype=np_int32)
	print("Number of batches: %d\nSource Vocabulary Size: %d" % (sum(ndl), nwordi,))

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
