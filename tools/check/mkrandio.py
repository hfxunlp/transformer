#encoding: utf-8

import sys
from numpy import array as np_array, int32 as np_int32
from numpy.random import randint as np_randint

from utils.h5serial import h5File

from cnfg.ihyp import h5_libver, h5datawargs

def handle(bsize, seql, nword, frs, ndata=1):

	_bsize = bsize
	with h5File(frs, "w", libver=h5_libver) as rsf:
		src_grp = rsf.create_group("src")
		tgt_grp = rsf.create_group("tgt")
		for curd in range(ndata):
			wid = str(curd)
			_size = (_bsize, seql,)
			src_grp.create_dataset(wid, data=np_randint(0, high=nword, size=_size, dtype=np_int32), **h5datawargs)
			tgt_grp.create_dataset(wid, data=np_randint(0, high=nword, size=_size, dtype=np_int32), **h5datawargs)
			_bsize += 1
			curd += 1
		rsf["ndata"] = np_array([ndata], dtype=np_int32)
		rsf["nword"] = np_array([nword, nword], dtype=np_int32)
	print("Number of batches: %d\nSource Vocabulary Size: %d\nTarget Vocabulary Size: %d" % (curd, nword, nword,))

if __name__ == "__main__":
	handle(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[-1])) if len(sys.argv) > 5 else handle(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
