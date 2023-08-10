#encoding: utf-8

""" usage:
	python tools/check/para.py $model_file.h5
"""

import sys
from h5py import Dataset

from utils.h5serial import h5File

def handle_group(srcg):

	rs = 0
	for k, v in srcg.items():
		if isinstance(v, Dataset):
			rs += v.size
		else:
			rs += handle_group(v)

	return rs

def handle(srcf):

	with h5File(srcf, "r") as sfg:
		rs = handle_group(sfg)
	print(rs)

if __name__ == "__main__":
	handle(sys.argv[1])
