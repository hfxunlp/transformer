#encoding: utf-8

import sys

from utils.fmt.vocab.base import merge_vocab
from utils.fmt.vocab.char import ldvocab_freq, save_vocab

def handle(srcfl, rsf, vsize=65532):

	save_vocab(merge_vocab(*[ldvocab_freq(_)[0] for _ in srcfl]), rsf, omit_vsize=vsize)

if __name__ == "__main__":
	handle(sys.argv[1:-2], sys.argv[-2], int(sys.argv[-1]))
