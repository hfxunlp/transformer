#encoding: utf-8

import sys
from transformers import MBartTokenizerFast as Tokenizer

from utils.fmt.plm.token import map_file as map_func

def handle(fsrc, vcb, frs, lang):

	return map_func(fsrc, frs, processor=Tokenizer(tokenizer_file=vcb, src_lang=lang))

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
