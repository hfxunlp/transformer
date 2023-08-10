#encoding: utf-8

import sys

from utils.fmt.base import clean_list, clean_list_iter, sys_open
from utils.fmt.vocab.token import save_vocab

def handle(srcfl, rsf, rslangf, vsize=65532):

	vocab = {}
	lang_vocab = {}

	curid = 0
	for srcf in srcfl:
		if srcf == "--target":
			break
		with sys_open(srcf, "rb") as f:
			for line in f:
				tmp = line.strip()
				if tmp:
					tokens = clean_list(tmp.decode("utf-8").split())
					for token in tokens[1:]:
						vocab[token] = vocab.get(token, 0) + 1
					token = tokens[0]
					lang_vocab[token] = lang_vocab.get(token, 0) + 1
		curid += 1

	for srcf in srcfl[curid+1:]:
		with sys_open(srcf, "rb") as f:
			for line in f:
				tmp = line.strip()
				if tmp:
					for token in clean_list_iter(tmp.decode("utf-8").split()):
						vocab[token] = vocab.get(token, 0) + 1

	save_vocab(vocab, rsf, omit_vsize=vsize)
	save_vocab(lang_vocab, rslangf, omit_vsize=False)

if __name__ == "__main__":
	handle(sys.argv[1:-3], sys.argv[-3], sys.argv[-2], int(sys.argv[-1]))
