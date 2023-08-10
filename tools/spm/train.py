#encoding: utf-8

# portal from fairseq: https://github.com/pytorch/fairseq/blob/master/scripts/spm_train.py

import sys

from sentencepiece SentencePieceTrainer

if __name__ == "__main__":
	SentencePieceTrainer.Train(" ".join(sys.argv[1:]))
