#encoding: utf-8

''' usage:
	python tools/average_model.py $averaged_model_file.t7 $model1.t7 $ model2.t7 ...
'''

import sys

import torch

rsm = torch.load(sys.argv[2], map_location='cpu')

nmodel = 1

for modelf in sys.argv[3:]:
	for basep, mpload in zip(rsm, torch.load(modelf, map_location='cpu')):
		basep.add_(mpload)
	nmodel += 1

nmodel = float(nmodel)

for basep in rsm:
	basep.div_(nmodel)

torch.save(rsm, sys.argv[1])
