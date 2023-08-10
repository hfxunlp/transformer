#encoding: utf-8

import sys

from utils.fmt.vocab.token import ldvocab

def handle(srcfl):

	for srcf in srcfl:
		print("The vocabulary size of %s is: %d (with special tokens counted)" % (srcf, ldvocab(srcf, minf=False, omit_vsize=False, vanilla=False)[-1],))

if __name__ == "__main__":
	handle(sys.argv[1:])
