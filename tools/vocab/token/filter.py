#encoding: utf-8

import sys

from utils.fmt.lang.zh.t2s import vcb_filter_func as filter_func
from utils.fmt.parser import parse_none
from utils.fmt.vocab.token import ldvocab_freq, save_vocab

def handle(srcf, rsf, vsize=65532, omit_vsize=None):

	save_vocab(filter_func(ldvocab_freq(srcf, omit_vsize=vsize)[0]), rsf, omit_vsize=parse_none(omit_vsize, vsize))

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
