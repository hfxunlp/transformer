#encoding: utf-8

''' usage:
	python tools/average_model.py $averaged_model_file.h5 $model1.h5 $model2.h5 ...
'''

import sys

import torch

from utils.base import secure_type_map
from utils.h5serial import h5save, h5load

from cnfg.ihyp import *

def handle(srcfl, rsf):

	rsm = h5load(srcfl[0])

	src_type = [para.dtype for para in rsm]
	map_type = [secure_type_map[para.dtype] if para.dtype in secure_type_map else None for para in rsm]
	sec_rsm = [para if typ is None else para.to(typ) for para, typ in zip(rsm, map_type)]

	nmodel = 1
	for modelf in srcfl[1:]:
		for basep, mpload, typ in zip(sec_rsm, h5load(modelf), map_type):
			basep.add_(mpload if typ is None else mpload.to(typ))
		nmodel += 1
	nmodel = float(nmodel)
	for basep in sec_rsm:
		basep.div_(nmodel)

	rsm = [para if mtyp is None else para.to(styp) for para, mtyp, styp in zip(sec_rsm, map_type, src_type)]

	h5save(rsm, rsf, h5args=h5zipargs)

if __name__ == "__main__":
	handle(sys.argv[2:], sys.argv[1])
