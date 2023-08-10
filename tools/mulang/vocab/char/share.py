#encoding: utf-8

import sys

from utils.fmt.base import sys_open
from utils.fmt.vocab.char import save_vocab

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
					tokens = tmp.decode("utf-8")
					_ = tokens.find(" ")
					for token in tokens[_ + 1:]:
						vocab[token] = vocab.get(token, 0) + 1
					token = tokens[:_]
					lang_vocab[token] = lang_vocab.get(token, 0) + 1
		curid += 1

	for srcf in srcfl[curid+1:]:
		with sys_open(srcf, "rb") as f:
			for line in f:
				tmp = line.strip()
				if tmp:
					for token in tmp.decode("utf-8"):
						vocab[token] = vocab.get(token, 0) + 1

	save_vocab(vocab, rsf, omit_vsize=vsize)
	save_vocab(lang_vocab, rslangf, omit_vsize=False)

if __name__ == "__main__":
	handle(sys.argv[1:-3], sys.argv[-3], sys.argv[-2], int(sys.argv[-1]))
