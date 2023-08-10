#encoding: utf-8

import sys

from utils.fmt.base import sys_open
from utils.fmt.vocab.char import save_vocab

def handle(srcfl, rsf, vsize=65532):

	vocab = {}

	for srcf in srcfl:
		with sys_open(srcf, "rb") as f:
			for line in f:
				tmp = line.strip()
				if tmp:
					for token in tmp.decode("utf-8"):
						vocab[token] = vocab.get(token, 0) + 1

	save_vocab(vocab, rsf, omit_vsize=vsize)

if __name__ == "__main__":
	handle(sys.argv[1:-2], sys.argv[-2], int(sys.argv[-1]))
