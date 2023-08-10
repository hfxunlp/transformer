#encoding: utf-8

import sys

from utils.fmt.base import clean_list, sys_open

def doc_reader(fname, sep=None):

	with sys_open(fname, "rb") as frd:
		cache = []
		max_tok = 0
		for line in frd:
			tmp = line.strip()
			if tmp:
				tmp = clean_list(tmp.decode("utf-8").split(sep=sep))
				_ld = len(tmp)
				if _ld > max_tok:
					max_tok = _ld
				cache.append(tmp)
			else:
				yield cache, max_tok
				cache = []
				max_tok = 0
		if cache:
			yield cache, max_tok

def legal_vocab(doc, ilgset, ratio):

	total = 0
	ilg = 0
	for sent in doc:
		for tmpu in sent.split():
			if tmpu:
				if tmpu in ilgset:
					ilg += 1
				total += 1
	rt = float(ilg) / float(total)

	return False if rt > ratio else True

def get_char_ratio(doc):

	ntokens = 0
	nchars = 0
	nsp = 0
	lorigin = 0
	for sent in doc:
		pbpe = False
		for tmpu in sent.split():
			if tmpu:
				if tmpu.endswith("@@"):
					nchars += 1
					if not pbpe:
						pbpe = True
						nsp += 1
				elif pbpe:
					pbpe = False
				ntokens += 1
		lorigin += len(sent.replace("@@ ", "").split())
	lorigin = float(lorigin)
	ntokens = float(ntokens)

	return float(nchars) / ntokens, ntokens / lorigin, float(nsp) / lorigin
