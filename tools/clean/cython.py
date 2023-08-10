#encoding: utf-8

import sys
from os import remove, walk
from os.path import join as pjoin

def walk_path(ptws):

	for ptw in ptws:
		for root, dirs, files in walk(ptw):
			for pyf in files:
				if pyf.endswith(".c") or pyf.endswith(".so"):
					_pyf = pjoin(root, pyf)
					remove(_pyf)

if __name__ == "__main__":
	walk_path(sys.argv[1:])
