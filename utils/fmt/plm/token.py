#encoding: utf-8

import sys

from utils.fmt.base import tostr

tokenize_line = lambda lin, processor: " ".join(processor.convert_ids_to_tokens(processor(lin, return_token_type_ids=True, return_attention_mask=False, return_offsets_mapping=True).input_ids))
map_line = lambda lin, processor: " ".join(tostr(processor(*lin.split("\t"), return_token_type_ids=True, return_attention_mask=False, return_offsets_mapping=True).input_ids))
detokenize_line = lambda lin, processor: processor(lin, skip_special_tokens=False, clean_up_tokenization_spaces=False)

def map_line_with_token_type(lin, processor):

	_ = processor(*tmp.decode("utf-8").split("\t"), return_token_type_ids=True, return_attention_mask=False, return_offsets_mapping=True)

	return " ".join(tostr(_.input_ids)), " ".join(tostr(_.token_type_ids))

def loop_file_so(fsrc, vcb, frs, process_func=None, processor=None):

	ens = "\n".encode("utf-8")
	with sys.stdin.buffer if fsrc == "-" else open(fsrc, "rb") as frd, sys.stdout.buffer if frs == "-" else open(frs, "wb") as fwrt:
		for line in frd:
			tmp = line.strip()
			if tmp:
				fwrt.write(process_func(tmp.decode("utf-8"), processor).encode("utf-8"))
			fwrt.write(ens)

def tokenize_file(fsrc, vcb, frs, Tokenizer=None):

	return loop_file_so(fsrc, vcb, frs, process_func=tokenize_line, processor=Tokenizer(tokenizer_file=vcb))

def map_file(fsrc, vcb, frs, Tokenizer=None):

	return loop_file_so(fsrc, vcb, frs, process_func=map_line, processor=Tokenizer(tokenizer_file=vcb))

def map_file_with_token_type(fsrc, vcb, frsi, frst, Tokenizer=None):

	tokenizer = Tokenizer(tokenizer_file=vcb)
	ens = "\n".encode("utf-8")
	with sys.stdin.buffer if fsrc == "-" else open(fsrc, "rb") as frd, sys.stdout.buffer if frsi == "-" else open(frsi, "wb") as fwrti, sys.stdout.buffer if frst == "-" else open(frst, "wb") as fwrtt:
		for line in frd:
			tmp = line.strip()
			if tmp:
				_input_ids, _token_type_ids = map_line_with_token_type(tmp.decode("utf-8"), tokenizer)
				fwrti.write(_input_ids.encode("utf-8"))
				fwrtt.write(_token_type_ids.encode("utf-8"))
			fwrti.write(ens)
			fwrtt.write(ens)

def map_back_file(fsrc, vcb, frs, Tokenizer=None):

	return loop_file_so(fsrc, vcb, frs, process_func=detokenize_line, processor=Tokenizer(tokenizer_file=vcb).decode)
