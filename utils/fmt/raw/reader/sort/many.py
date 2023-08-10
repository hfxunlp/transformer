#encoding: utf-8

from random import shuffle

from utils.fmt.base import dict_insert_set, iter_dict_sort, read_lines
from utils.fmt.parser import parse_none

def sort_list_reader(x, *args, clear_input=True, **kwargs):

	_d = {}
	for mi in x:
		lens = [len(_) for _ in mi]
		lgth = sum(lens)
		_d = dict_insert_set(_d, mi, lgth, *reversed(lens[1:]))
	if clear_input and hasattr(x, "clear"):
		x.clear()
	for tmp in iter_dict_sort(_d, free=True):
		_v = list(tmp)
		shuffle(_v)
		yield from _v

class sort_lines_reader:

	def __init__(self, line_read=None):

		self.line_read = line_read

	def __call__(self, x, *args, line_read=None, **kwargs):

		_line_read = parse_none(line_read, self.line_read)
		_data_iter = x if _line_read is None else read_lines(x, _line_read)
		_d = {}
		for mi in _data_iter:
			lens = [len(_) for _ in mi]
			lgth = sum(lens)
			_d = dict_insert_set(_d, mi, lgth, *reversed(lens[1:]))
		for tmp in iter_dict_sort(_d, free=True):
			_v = list(tmp)
			shuffle(_v)
			yield from _v
