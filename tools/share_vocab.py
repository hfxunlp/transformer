#encoding: utf-8

import sys

from utils.fmt.base import clean_list_iter, save_vocab

def handle(srcfl, rsf, vsize=65532):

	vocab = {}

	for srcf in srcfl:
		with open(srcf, "rb") as f:
			for line in f:
				tmp = line.strip()
				if tmp:
					for token in clean_list_iter(tmp.decode("utf-8").split()):
						vocab[token] = vocab.get(token, 0) + 1

	save_vocab(vocab, rsf, omit_vsize=vsize)

if __name__ == "__main__":
	handle(sys.argv[1:-2], sys.argv[-2], int(sys.argv[-1]))
