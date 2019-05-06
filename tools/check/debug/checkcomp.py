#encoding: utf-8

import sys

import torch

rsm = torch.load(sys.argv[1], map_location='cpu')

for k, v in rsm.items():
	print(k)
