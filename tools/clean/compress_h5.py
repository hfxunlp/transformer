#encoding: utf-8

import sys

from utils.h5serial import h5save, h5load

def handle(srcf, rsf):

	h5save(h5load(srcf, restore_list=False), rsf, h5args={"compression": "gzip", "compression_opts": 9, "shuffle":True})

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[-1])
