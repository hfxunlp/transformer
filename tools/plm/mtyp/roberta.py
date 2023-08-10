#encoding: utf-8

import sys
from transformers import RobertaTokenizerFast as Tokenizer

from utils.fmt.plm.token import map_file_with_token_type as map_func

def handle(fsrc, vcb, frsi, frst):

	return map_func(fsrc, frsi, frst, processor=Tokenizer(tokenizer_file=vcb))

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
