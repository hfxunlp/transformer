#encoding: utf-8

import sys

import numpy

from utils.h5serial import h5File
from utils.fmt.base import ldvocab
from utils.fmt.mulang.eff.dual import batch_padder

from cnfg.ihyp import *

def handle(finput, ftarget, fvocab_i, fvocab_t, fvocab_task, frs, minbsize=1, expand_for_mulgpu=True, bsize=max_sentences_gpu, maxpad=max_pad_tokens_sentence, maxpart=normal_tokens_vs_pad_tokens, maxtoken=max_tokens_gpu, minfreq=False, vsize=False):
	vcbi, nwordi = ldvocab(fvocab_i, minf=minfreq, omit_vsize=vsize, vanilla=False)
	vcbt, nwordt = ldvocab(fvocab_t, minf=minfreq, omit_vsize=vsize, vanilla=False)
	vcbtask, nwordtask = ldvocab(fvocab_task, minf=False, omit_vsize=False, vanilla=True)
	if expand_for_mulgpu:
		_bsize = bsize * minbsize
		_maxtoken = maxtoken * minbsize
	else:
		_bsize = bsize
		_maxtoken = maxtoken
	with h5File(frs, 'w') as rsf:
		curd = {}
		torder = []
		for i_d, td, taskd in batch_padder(finput, ftarget, vcbi, vcbt, vcbtask, _bsize, maxpad, maxpart, _maxtoken, minbsize):
			_str_taskd = str(taskd)
			if _str_taskd in rsf:
				task_grp = rsf[_str_taskd]
				src_grp = task_grp["src"]
				tgt_grp = task_grp["tgt"]
			else:
				task_grp = rsf.create_group(_str_taskd)
				src_grp = task_grp.create_group("src")
				tgt_grp = task_grp.create_group("tgt")
				torder.append(taskd)
			rid = numpy.array(i_d, dtype=numpy.int32)
			rtd = numpy.array(td, dtype=numpy.int32)
			_id = curd.get(taskd, 0)
			wid = str(_id)
			src_grp.create_dataset(wid, data=rid, **h5datawargs)
			tgt_grp.create_dataset(wid, data=rtd, **h5datawargs)
			curd[taskd] = _id + 1
		rsf["taskorder"] = numpy.array(torder, dtype=numpy.int32)
		curd = [curd[tmp] for tmp in torder]
		rsf["ndata"] = numpy.array(curd, dtype=numpy.int32)
		rsf["nword"] = numpy.array([nwordi, nwordtask, nwordt], dtype=numpy.int32)
	print("Number of Batches: %d\nSource Vocabulary Size: %d\nTarget Vocabulary Size: %d\nNumber of Tasks: %d" % (sum(curd), nwordi, nwordt, nwordtask,))

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], int(sys.argv[7]))
