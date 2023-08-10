#encoding: utf-8

from itertools import accumulate
from math import log

def pos_norm(x):

	_s = sum(x)
	if _s == 0.0:
		_s = 1.0

	return [_ / _s for _ in x]

def cumsum(*args, **kwargs):

	return list(accumulate(*args, **kwargs))

def arcsigmoid(x):

	return -log((1.0 / x) - 1.0)

def exp_grow(start, end, k):

	_ng = k - 1
	_factor = (end / start) ** (1.0 / _ng)
	tmp = start
	rs = [start]
	for i in range(_ng):
		tmp *= _factor
		rs.append(tmp)

	return rs

def linear_grow(start, end, k):

	_ng = k - 1
	_factor = (end - start) / _ng
	tmp = start
	rs = [start]
	for i in range(_ng):
		tmp += _factor
		rs.append(tmp)

	return rs

def comb_grow(start, end, k, alpha=0.5):

	beta = 1.0 - alpha

	return [a * alpha + b * beta for a, b in zip(exp_grow(start, end, k), linear_grow(start, end, k))]
