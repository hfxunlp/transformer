#encoding: utf-8

from random import random

def multinomial(lin, s=None):

	_s = sum(lin) if s is None else s
	_p = random()
	if _s != 1.0:
		_p *= _s
	rs_ind = len(lin) - 1
	for i, lu in enumerate(lin):
		_p -= lu
		if _p <= 0.0:
			rs_ind = i
			break

	return rs_ind

def multinomial_k(lin, k, s=None):

	_s = sum(lin) if s is None else s
	rs = []
	init_rs_ind = len(lin) - 1
	for i in range(k):
		_p = random()
		if _s != 1.0:
			_p *= _s
		rs_ind = init_rs_ind
		for i, lu in enumerate(lin):
			_p -= lu
			if _p <= 0.0:
				rs_ind = i
				break
		rs.append(rs_ind)

	return rs
