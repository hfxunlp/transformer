#encoding: utf-8

import sys

import h5py

from cnfg.ihyp import *

def handle_group(srcg, rsg, h5args=h5zipargs):

	for k, v in srcg.items():
		if isinstance(v, h5py.Dataset):
			rsg.create_dataset(k, data=v[:], **h5args)
		else:
			rsg.create_group(k)
			handle_group(v, rsg[k], h5args=h5args)

def handle(srcf, rsf, h5args=h5zipargs):

	sfg, rfg = h5py.File(srcf, "r"), h5py.File(rsf, 'w')
	handle_group(sfg, rfg, h5args=h5args)
	sfg.close()
	rfg.close()

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[-1])
