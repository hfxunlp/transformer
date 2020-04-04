#encoding: utf-8

import sys

import torch
from utils.h5serial import h5save, h5load

from cnfg.ihyp import *

def torch_to_h5(srcf, rsf, h5args=h5zipargs):
	h5save(torch.load(srcf, map_location='cpu'), rsf, h5args=h5args)

def h5_to_torch(srcf, rsf):
	torch.save(h5load.load(srcf, restore_list=True), rsf)

def handle(srcf, rsf):

	_execf = h5_to_torch if srcf.endswith(".h5") else torch_to_h5
	_execf(srcf, rsf)

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[-1])
