#encoding: utf-8

import sys

from utils.fmt.base import init_normal_token_id, clean_list_iter

def handle(srcfl):

	global init_normal_token_id

	vocab = set()

	for srcf in srcfl:
		with open(srcf, "rb") as f:
			for line in f:
				tmp = line.strip()
				if tmp:
					for token in clean_list_iter(tmp.decode("utf-8").split()):
						if not token in vocab:
							vocab.add(token)

	nvcb = len(vocab)
	nvcb += init_normal_token_id

	print("The size of the vocabulary is: %d (with special tokens counted)" % (nvcb))

if __name__ == "__main__":
	handle(sys.argv[1:])
