#encoding: utf-8

import sys
from transformers import BertTokenizerFast as Tokenizer

from utils.fmt.plm.token import tokenize_file as map_func

def handle(fsrc, vcb, frs):

	return map_func(fsrc, frs, processor=Tokenizer(tokenizer_file=vcb))

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3])
