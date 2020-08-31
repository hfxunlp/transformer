#encoding: utf-8

''' usage:
	python tools/check/para.py $model_file.h5
'''

import sys

import h5py

def handle_group(srcg):

	rs = 0
	for k, v in srcg.items():
		if isinstance(v, h5py.Dataset):
			rs += v[:].size
		else:
			rs += handle_group(v)

	return rs

def handle(srcf):

	sfg = h5py.File(srcf, "r")
	rs = handle_group(sfg)
	sfg.close()
	print(rs)

if __name__ == "__main__":
	handle(sys.argv[1])
