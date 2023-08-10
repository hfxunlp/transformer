#encoding: utf-8

import sys
from h5py import Dataset

from utils.h5serial import h5File, h5load, h5save

from cnfg.ihyp import *

def handle_group(srcg, rsg, h5args=h5zipargs):

	for k, v in srcg.items():
		if isinstance(v, Dataset):
			rsg.create_dataset(k, data=v[()], **h5args)
		else:
			rsg.create_group(k)
			handle_group(v, rsg[k], h5args=h5args)

def handle(srcf, rsf, h5args=h5zipargs):

	if srcf == rsf:
		h5save(h5load(srcf, restore_list=False), rsf, h5args=h5args)
	else:
		with h5File(srcf, "r") as sfg, h5File(rsf, "w", libver=h5_libver) as rfg:
			handle_group(sfg, rfg, h5args=h5args)

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[-1])
