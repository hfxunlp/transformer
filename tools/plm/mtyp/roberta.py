#encoding: utf-8

import sys
from transformers import RobertaTokenizerFast as Tokenizer

from utils.fmt.plm.token import map_file_with_token_type as map_func

def handle(*inputs, **kwargs):

	return map_func(*inputs, **kwargs, Tokenizer=Tokenizer)

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
