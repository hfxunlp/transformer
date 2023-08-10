#encoding: utf-8

from difflib import SequenceMatcher

def seq_diff(x, ref):

	for tag, xsi, xei, rsi, rei in SequenceMatcher(None, x, ref, autojunk=False).get_opcodes():
		_tc = tag[0]
		if _tc == "d":
			for _ in x[xsi:xei]:
				yield _tc, _
		elif _tc == "e":
			for _ in x[xsi:xei]:
				yield _tc, _
		elif _tc == "i":
			for _ in ref[rsi:rei]:
				yield _tc, _
		else:
			for _ in x[xsi:xei]:
				yield "d", _
			for _ in ref[rsi:rei]:
				yield "i", _

def reorder_insert(seqin):

	_d_cache = []
	for _du in seqin:
		_op = _du[0]
		if _op == "d":
			_d_cache.append(_du)
		else:
			if (_op == "e") and _d_cache:
				yield from _d_cache
				_d_cache = []
			yield _du
	if _d_cache:
		yield from _d_cache

def seq_diff_reorder_insert(x, ref):

	for tag, xsi, xei, rsi, rei in SequenceMatcher(None, x, ref, autojunk=False).get_opcodes():
		_tc = tag[0]
		if _tc == "d":
			for _ in x[xsi:xei]:
				yield _tc, _
		elif _tc == "e":
			for _ in x[xsi:xei]:
				yield _tc, _
		elif _tc == "i":
			for _ in ref[rsi:rei]:
				yield _tc, _
		else:
			for _ in ref[rsi:rei]:
				yield "i", _
			for _ in x[xsi:xei]:
				yield "d", _

seq_diff_ratio = lambda x, ref: SequenceMatcher(None, x, ref, autojunk=False).ratio()
seq_diff_ratio_ub = lambda x, ref: SequenceMatcher(None, x, ref, autojunk=False).quick_ratio()
seq_diff_ratio_ub_fast = lambda x, ref: SequenceMatcher(None, x, ref, autojunk=False).real_quick_ratio()
