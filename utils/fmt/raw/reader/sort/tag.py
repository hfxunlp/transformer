#encoding: utf-8

from random import shuffle

from utils.fmt.base import read_lines
from utils.fmt.parser import parse_none

def sort_list_reader(x, *args, clear_input=True, **kwargs):

	_d = {}
	for _ in x:
		_k = len(_[0])
		if _k in _d:
			if _ not in _d[_k]:
				_d[_k].add(_)
		else:
			_d[_k] = set([_])
	if clear_input and hasattr(x, "clear"):
		x.clear()
	for _k in sorted(_d.keys()):
		_v = list(_d.pop(_k))
		shuffle(_v)
		yield from _v

class sort_lines_reader:

	def __init__(self, line_read=None):

		self.line_read = line_read

	def __call__(self, x, *args, line_read=None, **kwargs):

		_line_read = parse_none(line_read, self.line_read)
		_data_iter = x if _line_read is None else read_lines(x, _line_read)
		_d = {}
		for _ in _data_iter:
			_k = len(_[0])
			if _k in _d:
				if _ not in _d[_k]:
					_d[_k].add(_)
			else:
				_d[_k] = set([_])
		for _k in sorted(_d.keys()):
			_v = list(_d.pop(_k))
			shuffle(_v)
			yield from _v
