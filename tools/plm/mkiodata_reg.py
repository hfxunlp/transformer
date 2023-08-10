#encoding: utf-8

import sys
from numpy import array as np_array, float32 as np_float32, int32 as np_int32

# import batch_padder of the corresponding model for different padding indices.

from utils.fmt.plm.roberta.dual_reg import batch_padder
from utils.h5serial import h5File

from cnfg.ihyp import *

def handle(finput, ftarget, frs, minbsize=1, expand_for_mulgpu=True, bsize=max_sentences_gpu, maxpad=max_pad_tokens_sentence, maxpart=normal_tokens_vs_pad_tokens, maxtoken=max_tokens_gpu, minfreq=False, vsize=False):

	if expand_for_mulgpu:
		_bsize = bsize * minbsize
		_maxtoken = maxtoken * minbsize
	else:
		_bsize = bsize
		_maxtoken = maxtoken
	with h5File(frs, "w", libver=h5_libver) as rsf:
		src_grp = rsf.create_group("src")
		tgt_grp = rsf.create_group("tgt")
		curd = 0
		for i_d, td in batch_padder(finput, ftarget, _bsize, maxpad, maxpart, _maxtoken, minbsize):
			rid = np_array(i_d, dtype=np_int32)
			rtd = np_array(td, dtype=np_float32)
			wid = str(curd)
			src_grp.create_dataset(wid, data=rid, **h5datawargs)
			tgt_grp.create_dataset(wid, data=rtd, **h5datawargs)
			curd += 1
		rsf["ndata"] = np_array([curd], dtype=np_int32)
	print("Number of batches: %d" % curd)

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
