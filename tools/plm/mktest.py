#encoding: utf-8

import sys
from numpy import array as np_array, int32 as np_int32

# import batch_padder of the corresponding model for different padding indices.
from utils.fmt.plm.roberta.single import batch_padder
from utils.h5serial import h5File

from cnfg.ihyp import *

def handle(finput, frs, minbsize=1, expand_for_mulgpu=True, bsize=max_sentences_gpu, maxpad=max_pad_tokens_sentence, maxpart=normal_tokens_vs_pad_tokens, maxtoken=max_tokens_gpu, minfreq=False, vsize=False):

	if expand_for_mulgpu:
		_bsize = bsize * minbsize
		_maxtoken = maxtoken * minbsize
	else:
		_bsize = bsize
		_maxtoken = maxtoken
	with h5File(frs, "w", libver=h5_libver) as rsf:
		src_grp = rsf.create_group("src")
		curd = 0
		for i_d in batch_padder(finput, _bsize, maxpad, maxpart, _maxtoken, minbsize):
			rid = np_array(i_d, dtype=np_int32)
			src_grp.create_dataset(str(curd), data=rid, **h5datawargs)
			curd += 1
		rsf["ndata"] = np_array([curd], dtype=np_int32)
	print("Number of batches: %d" % curd)

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], int(sys.argv[3]))
