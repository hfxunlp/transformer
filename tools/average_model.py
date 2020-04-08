#encoding: utf-8

''' usage:
	python tools/average_model.py $averaged_model_file.h5 $model1.h5 $ model2.h5 ...
'''

import sys

import torch

from utils.h5serial import h5save, h5load

from cnfg.ihyp import *

def handle(srcfl, rsf):

	rsm = h5load(srcfl[0])
	nmodel = 1
	for modelf in srcfl[1:]:
		for basep, mpload in zip(rsm, h5load(modelf)):
			basep.add_(mpload)
		nmodel += 1
	nmodel = float(nmodel)
	for basep in rsm:
		basep.div_(nmodel)
	h5save(rsm, rsf, h5args=h5zipargs)

if __name__ == "__main__":
	handle(sys.argv[2:], sys.argv[1])
