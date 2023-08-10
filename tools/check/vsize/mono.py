#encoding: utf-8

import sys

from utils.fmt.base import clean_list_iter, sys_open

def handle(srcfl):

	vocab = set()

	for srcf in srcfl:
		with sys_open(srcf, "rb") as f:
			for line in f:
				tmp = line.strip()
				if tmp:
					for token in clean_list_iter(tmp.decode("utf-8").split()):
						if not token in vocab:
							vocab.add(token)

	nvcb = len(vocab)

	print("The size of the vocabulary is: %d (special tokens discounted)" % (nvcb))

if __name__ == "__main__":
	handle(sys.argv[1:])
