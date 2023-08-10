#encoding: utf-8

# PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python is required to load spm files with protobuf when tokenizer.json is not available.

import sys
from transformers import MBart50TokenizerFast as Tokenizer

from utils.fmt.plm.token import map_file as map_func

def handle(fsrc, vcb, frs, lang):

	return map_func(fsrc, frs, processor=Tokenizer.from_pretrained(vcb, src_lang=lang))

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
